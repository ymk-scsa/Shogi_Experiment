import sys
import os
import time
import argparse
import logging
import signal
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from collections import defaultdict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import create_model
from game.board import FEATURES_SETTINGS, FEATURES_NUM, MOVE_LABELS_NUM
from data.buffer import HcpeDataLoader, PsvDataLoader

# ===== ロガー設定 =====
def _get_logger(log_file: str = None, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_file:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

# ===== 精度・損失計算 =====

def accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    pred = ((y + 1.0) / 2.0) >= 0.5
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d)

def save_checkpoint(path: str, model, optimizer, epoch: int, step: int) -> None:
    _ensure_dir(path)
    state_dict = model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "t": step,
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

def save_loss_graph(history, mode: str, output_dir="data/"):
    if not HAS_MATPLOTLIB:
        return
    _ensure_dir(output_dir)
    data = history[mode]
    if not data["step"]: return
    
    plt.figure(figsize=(10, 6))
    plt.plot(data["step"], data["p_loss"], label="Policy Loss")
    plt.plot(data["step"], data["v_loss"], label="Value Loss")
    plt.plot(data["step"], data["t_loss"], label="Total Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - {mode}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_{mode}.png"))
    plt.close()

# ===== ワーカープロセス =====

def train_worker(
    rank: int,
    num_gpus: int,
    args: argparse.Namespace,
    model_list: List[str]
):
    """
    1つのモデルを担当して学習を行うプロセス。
    """
    model_mode = model_list[rank]
    
    # デバイス割り当て (GPUがあれば分散、なければCPU)
    if num_gpus > 0:
        device_id = rank % num_gpus
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    # ログ設定
    log_file = os.path.join(args.log_dir, f"train_{model_mode}.log")
    _ensure_dir(log_file)
    logger = _get_logger(log_file, name=model_mode)
    
    logger.info("Starting worker for %s on device %s", model_mode, device)

    # 特徴量設定
    features_setting = FEATURES_SETTINGS[args.input_features]
    input_ch = features_setting.features_num
    
    # モデル構築
    model = create_model(input_channels=input_ch, num_actions=MOVE_LABELS_NUM, mode=model_mode)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # データロード
    logger.info("Loading training data (Format: %s)...", args.format)
    LoaderClass = PsvDataLoader if args.format == "psv" else HcpeDataLoader
    
    train_loader = LoaderClass(
        args.train_data, args.batch, device, shuffle=True, features_mode=args.input_features
    )
    test_loader = LoaderClass(
        args.test_data, args.test_batch, device, features_mode=args.input_features
    )

    # 再開
    current_step = 0
    start_epoch = 0
    if args.resume:
        # モデル名が含まれるチェックポイントを探す
        resume_file = args.resume.replace("{model}", model_mode)
        if os.path.exists(resume_file):
            ckpt = torch.load(resume_file, map_location=device)
            start_epoch = ckpt.get("epoch", 0)
            current_step = ckpt.get("t", 0)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("Resumed from %s", resume_file)
    
    history = defaultdict(lambda: {"step": [], "p_loss": [], "v_loss": [], "t_loss": []})
    
    # 信号ハンドラ (Ctrl+C対応)
    stop_requested = False
    def handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, handler)

    # NLLLoss and MSELoss
    nll_loss = nn.NLLLoss()
    mse_loss = nn.MSELoss()

    # 学習ループ
    for epoch in range(start_epoch, args.epochs):
        epoch_idx = epoch + 1
        model.train()
        
        running_p_loss = 0.0
        running_v_loss = 0.0
        steps_interval = 0
        
        logger.info("--- Starting Epoch %d ---", epoch_idx)
        
        for x, move_label, result in train_loader:
            if stop_requested: break
            
            log_policy, value = model(x)
            
            loss_p = nll_loss(log_policy, move_label)
            loss_v = mse_loss((value + 1.0) / 2.0, result)
            loss = loss_p + loss_v
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_step += 1
            steps_interval += 1
            running_p_loss += loss_p.item()
            running_v_loss += loss_v.item()
            
            if current_step % args.eval_interval == 0:
                avg_p = running_p_loss / max(1, steps_interval)
                avg_v = running_v_loss / max(1, steps_interval)
                
                # バリデーション
                model.eval()
                with torch.no_grad():
                    tx, tml, tres = test_loader.sample()
                    lp_t, v_t = model(tx)
                    test_pa = accuracy(lp_t, tml)
                    test_va = binary_accuracy(v_t, tres)
                
                logger.info("Epoch %d Step %d | Train Loss(P/V): %.4f/%.4f | Test Acc(P/V): %.4f/%.4f", 
                            epoch_idx, current_step, avg_p, avg_v, test_pa, test_va)
                
                history[model_mode]["step"].append(current_step)
                history[model_mode]["p_loss"].append(avg_p)
                history[model_mode]["v_loss"].append(avg_v)
                history[model_mode]["t_loss"].append(avg_p + avg_v)
                save_loss_graph(history, model_mode, output_dir=args.log_dir)
                
                running_p_loss = 0.0
                running_v_loss = 0.0
                steps_interval = 0
                model.train()

        # エポック終了時の保存
        cp_name = f"parallel-{model_mode}-ep{epoch_idx:03}.pth"
        save_checkpoint(os.path.join(args.checkpoint_dir, cp_name), model, optimizer, epoch_idx, current_step)
        logger.info("Epoch %d finished. Checkpoint saved: %s", epoch_idx, cp_name)
        
        if stop_requested:
            logger.info("Interrupt received. Saving interrupted state...")
            save_checkpoint(os.path.join(args.checkpoint_dir, f"interrupted-{model_mode}.pth"), model, optimizer, epoch_idx, current_step)
            break

# ===== メイン処理 =====

def main():
    parser = argparse.ArgumentParser(description="Parallel training for modelA-modelH")
    parser.add_argument("--train-data", nargs="+", required=True, help="Path to training data")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--batch", type=int, default=1024, choices=[256, 1024, 4096], help="Batch size")
    parser.add_argument("--test-batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", default="weights_parallel/")
    parser.add_argument("--log-dir", default="data_parallel/")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--input-features", type=int, default=0)
    parser.add_argument("--format", default="hcpe", choices=["hcpe", "psv"])
    parser.add_argument("--resume", default="", help="Resume path template with {model}")
    
    args = parser.parse_args()
    
    # 8つのモデルを定義
    model_list = ["modelA", "modelB", "modelC", "modelD", "modelE", "modelF", "modelG", "modelH"]
    num_workers = len(model_list)
    
    # GPU数の確認
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Spawning {num_workers} parallel workers.")
    
    # フォルダ準備
    _ensure_dir(args.checkpoint_dir)
    _ensure_dir(args.log_dir)

    # マルチプロセス起動
    mp.spawn(train_worker, nprocs=num_workers, args=(num_gpus, args, model_list))

if __name__ == "__main__":
    # Windows のマルチプロセス対応
    mp.set_start_method('spawn', force=True)
    main()
