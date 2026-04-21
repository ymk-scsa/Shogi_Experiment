"""
parallelization_train.py
========================

modelA〜modelH の 8 モデルを、それぞれ独立ワーカープロセスで並列学習するスクリプト。

主な機能:
- モデルごとの独立ログ出力
- モデルごとの checkpoint 保存
- 評価間隔ごとの損失/精度記録
- 損失グラフ保存（matplotlib がある場合のみ）
"""

import sys
import os
import argparse
import logging
import signal
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import create_model
from game.board import FEATURES_SETTINGS, MOVE_LABELS_NUM
from data.past_buffer import HcpeDataLoader, PsvDataLoader

# ===== ロガー設定 =====
def _get_logger(log_file: str = None, name: str = "train") -> logging.Logger:
    """
    学習用ロガーを作成して返す。

    Args:
        log_file: ログファイルパス。未指定時は標準出力のみ。
        name: ロガー名（モデルごとに分けるために使用）。

    Returns:
        設定済みの Logger。
    """
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
    """方策出力の top-1 正解率を返す。"""
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    """
    価値出力の二値正解率を返す。

    Notes:
        model の value は tanh 出力 [-1, 1] を想定しているため、
        (y + 1) / 2 で [0, 1] に変換して閾値 0.5 で判定する。
    """
    pred = ((y + 1.0) / 2.0) >= 0.5
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)

# ===== ファイル保存 =====
def _ensure_dir(path: str) -> None:
    """
    親ディレクトリが存在しなければ作成する。

    Args:
        path: ファイルパスまたはディレクトリパス。
    """
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d)

def save_checkpoint(path: str, model, optimizer, epoch: int, step: int) -> None:
    """
    学習再開に必要な状態を checkpoint として保存する。

    Args:
        path: 保存先パス。
        model: 学習中モデル。
        optimizer: 対応する optimizer。
        epoch: 現在エポック。
        step: 現在ステップ。
    """
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
    """
    指定モデルの損失推移グラフを保存する。

    Args:
        history: 損失履歴辞書。
        mode: モデル名（例: modelA）。
        output_dir: 画像出力先ディレクトリ。
    """
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
    1ワーカープロセスで 1 モデルを担当して学習する。

    Args:
        rank: ワーカー番号（model_list のインデックスとして使用）。
        num_gpus: 利用可能 GPU 数。
        args: CLI 引数。
        model_list: 学習対象モデル名の配列。
    """
    model_mode = model_list[rank]
    
    # デバイス割り当て (GPUがあれば分散、なければCPU)
    if num_gpus > 0:
        device_id = rank % num_gpus
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    # ログはモデルごとに分離し、run_tag を付けてセッション衝突を避ける。
    log_suffix = f"_{args.run_tag}" if args.run_tag else ""
    log_file = os.path.join(args.log_dir, f"train_{model_mode}{log_suffix}.log")
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

    # チェックポイント再開。--resume には "{model}" プレースホルダを使える。
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
    
    # Ctrl+C を受けたら現在エポックの終わりで安全に保存して停止する。
    stop_requested = False
    def handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, handler)

    # Policy は生logitsを返すため CrossEntropyLoss を使う。
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # ===== 学習ループ =====
    for epoch in range(start_epoch, args.epochs):
        epoch_idx = epoch + 1
        model.train()
        
        running_p_loss = 0.0
        running_v_loss = 0.0
        steps_interval = 0
        
        logger.info("--- Starting Epoch %d ---", epoch_idx)
        
        for x, move_label, result in train_loader:
            if stop_requested: break
            
            # 補助出力は使わないため、返り値は policy/value の2要素に固定する。
            policy_logits, value = model(x, return_aux=False)
            
            loss_p = ce_loss(policy_logits, move_label)
            loss_v = mse_loss((value + 1.0) / 2.0, result)
            loss = loss_p + loss_v
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_step += 1
            steps_interval += 1
            running_p_loss += loss_p.item()
            running_v_loss += loss_v.item()
            
            # 一定ステップごとに検証して、ログと損失グラフを更新する。
            if current_step % args.eval_interval == 0:
                avg_p = running_p_loss / max(1, steps_interval)
                avg_v = running_v_loss / max(1, steps_interval)
                
                # バリデーション
                model.eval()
                test_pa_sum = 0.0
                test_va_sum = 0.0
                with torch.no_grad():
                    for _ in range(args.eval_batches):
                        tx, tml, tres = test_loader.sample()
                        p_t, v_t = model(tx, return_aux=False)
                        test_pa_sum += accuracy(p_t, tml)
                        test_va_sum += binary_accuracy(v_t, tres)
                test_pa = test_pa_sum / max(1, args.eval_batches)
                test_va = test_va_sum / max(1, args.eval_batches)
                
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

        # エポックごとに必ず checkpoint を保存する。
        cp_suffix = f"-{args.run_tag}" if args.run_tag else ""
        cp_name = f"parallel-{model_mode}-ep{epoch_idx:03}{cp_suffix}.pth"
        save_checkpoint(os.path.join(args.checkpoint_dir, cp_name), model, optimizer, epoch_idx, current_step)
        logger.info("Epoch %d finished. Checkpoint saved: %s", epoch_idx, cp_name)
        
        if stop_requested:
            logger.info("Interrupt received. Saving interrupted state...")
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"interrupted-{model_mode}{cp_suffix}.pth"),
                model,
                optimizer,
                epoch_idx,
                current_step,
            )
            break

# ===== メイン処理 =====

def main():
    """
    エントリーポイント。

    Notes:
        modelA〜modelH の 8 ワーカーを mp.spawn で起動し、
        各ワーカーが独立して 1 モデルずつ学習する。
    """
    parser = argparse.ArgumentParser(description="Parallel training for modelA-modelH")
    parser.add_argument("--train-data", nargs="+", required=True, help="Path to training data")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--batch", type=int, default=1024, choices=[256, 1024, 2048, 4096], help="Batch size")
    parser.add_argument("--test-batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", default="weights_parallel/")
    parser.add_argument("--log-dir", default="data_parallel/")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4, help="Number of random test batches per evaluation")
    parser.add_argument("--input-features", type=int, default=0)
    parser.add_argument("--format", default="hcpe", choices=["hcpe", "psv"])
    parser.add_argument("--resume", default="", help="Resume path template with {model}")
    parser.add_argument("--models", default="modelA,modelB,modelC,modelD,modelE,modelF,modelG,modelH",
                        help="Comma-separated model list (e.g. fastA,modelB)")
    parser.add_argument("--run-tag", default="", help="Optional suffix for log/checkpoint files")
    
    args = parser.parse_args()
    
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")

    # run-tag 未指定時はタイムスタンプを使ってログ衝突を避ける。
    if not args.run_tag:
        args.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 学習対象モデル一覧
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        raise ValueError("No models specified. Please set --models.")
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
