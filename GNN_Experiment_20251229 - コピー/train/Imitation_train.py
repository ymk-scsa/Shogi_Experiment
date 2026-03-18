import sys
import os
import time
import argparse
import logging
import signal
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import create_model
from game.board import FEATURES_SETTINGS, FEATURES_NUM, MOVE_LABELS_NUM
from data.buffer import HcpeDataLoader, PsvDataLoader

# ===== ロガー設定 =====
def _get_logger(log_file: str = None, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger(f"train_rank_{rank}")
    if not logger.hasHandlers():
        formatter = logging.Formatter(f"[%(asctime)s] [Rank {rank}] [%(levelname)s] %(name)s: %(message)s")
        # 標準出力への出力設定
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # ランク0の場合のみファイルに出力
        if log_file and rank == 0:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        logger.propagate = False
    return logger

# ===== PER (Prioritized Experience Replay) 関連 =====

class PrioritySampler:
    """
    優先度に基づいたサンプリングを管理するクラス。
    """
    def __init__(self, data_size: int, alpha: float = 0.6):
        self.data_size = data_size
        self.alpha = alpha
        # 初期優先度はすべて1.0
        self.priorities = np.ones(data_size, dtype=np.float32)
        self.indices = np.arange(data_size)

    def update_priorities(self, batch_indices: np.ndarray, errors: np.ndarray):
        """損失（誤差）に基づいて優先度を更新する。"""
        # 優先度は (error + epsilon) ^ alpha
        new_priorities = (np.abs(errors) + 1e-6) ** self.alpha
        self.priorities[batch_indices] = new_priorities

    def sample(self, batch_size: int) -> np.ndarray:
        """優先度に基づく確率分布に従ってインデックスをサンプリング。"""
        probs = self.priorities / self.priorities.sum()
        return np.random.choice(self.indices, size=batch_size, p=probs, replace=False)

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

def save_checkpoint(path: str, model, optimizer, epoch: int, step: int, rank: int) -> None:
    if rank != 0:
        return
    _ensure_dir(path)
    # DDPの場合は module を取り出す
    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "t": step,
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

def save_loss_graph(history, output_dir="data/", rank: int = 0):
    if rank != 0:
        return
    _ensure_dir(output_dir)
    for mode, data in history.items():
        if not data["step"]: continue
        plt.figure(figsize=(10, 6))
        plt.plot(data["step"], data["p_loss"], label="Policy Loss")
        plt.plot(data["step"], data["v_loss"], label="Value Loss")
        plt.plot(data["step"], data["t_loss"], label="Total Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Imitation Learning Loss - {mode}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"imitation_loss_{mode}.png"))
        plt.close()

# ===== DDP セットアップ =====

def setup_dist(rank, world_size, backend="gloo"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_dist():
    dist.destroy_process_group()

# ===== メイン学習ループ =====

def train_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    model_configs: List[Tuple[str, int]]
):
    # DDP 初期化
    if world_size > 1:
        setup_dist(rank, world_size, backend=args.backend)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and rank >= 0 else "cpu")
    logger = _get_logger(args.log, rank)
    
    if rank == 0:
        logger.info("=== Starting Imitation Learning Training ===")
        logger.info("Backend=%s, World_size=%d, Mode=%s", args.backend, world_size, args.models)

    # 特徴量設定
    features_setting = FEATURES_SETTINGS[args.input_features]
    input_ch = features_setting.features_num
    
    # 複数モデル対応だが、基本は1つのターゲット
    for mode, target_epochs in model_configs:
        model = create_model(input_channels=input_ch, num_actions=MOVE_LABELS_NUM, mode=mode)
        model.to(device)
        
        if world_size > 1:
            # DistributedDataParallel
            model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        # データロード
        if rank == 0: logger.info("Loading training data (Format: %s)...", args.format)
        LoaderClass = PsvDataLoader if args.format == "psv" else HcpeDataLoader
        
        train_loader = LoaderClass(
            args.train_data, args.batch, device, shuffle=False, features_mode=args.input_features
        )
        test_loader = LoaderClass(
            args.test_data, args.test_batch, device, features_mode=args.input_features
        )

        # PER 設定
        sampler = None
        if args.use_per:
            sampler = PrioritySampler(len(train_loader.data), alpha=0.6)
            if rank == 0: logger.info("Prioritized Experience Replay (PER) enabled.")

        # 再開
        current_step = 0
        start_epoch = 0
        if args.resume and rank == 0:
            if os.path.exists(args.resume):
                ckpt = torch.load(args.resume, map_location=device)
                start_epoch = ckpt.get("epoch", 0)
                current_step = ckpt.get("t", 0)
                inner_model = model.module if isinstance(model, DDP) else model
                inner_model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                logger.info("Resumed from %s", args.resume)

        history = defaultdict(lambda: {"step": [], "p_loss": [], "v_loss": [], "t_loss": []})
        
        # 信号ハンドラ
        stop_requested = False
        def handler(signum, frame):
            nonlocal stop_requested
            stop_requested = True
        signal.signal(signal.SIGINT, handler)

        nll_loss = nn.NLLLoss(reduction='none' if args.use_per else 'mean')
        mse_loss = nn.MSELoss(reduction='none' if args.use_per else 'mean')

        # 学習ループ
        for epoch in range(start_epoch, target_epochs):
            epoch_idx = epoch + 1
            model.train()
            
            # 各ランクが均等にステップを踏む
            steps_per_epoch = len(train_loader.data) // (args.batch * world_size)
            
            running_p_loss = 0.0
            running_v_loss = 0.0
            
            for step in range(steps_per_epoch):
                if stop_requested: break
                
                # サンプリング
                if args.use_per:
                    batch_indices = sampler.sample(args.batch)
                    # ランクごとに異なるインデックスを割り当てる場合は、ここで rank を考慮する
                    # 簡易化のため、各ランクで独立に PER サンプリングを行う
                    batch_data = train_loader.data[batch_indices]
                    x, move_label, result = train_loader.mini_batch(batch_data)
                else:
                    x, move_label, result = train_loader.sample()
                    batch_indices = None

                log_policy, value = model(x)
                
                # 損失計算
                loss_p_raw = nll_loss(log_policy, move_label)
                loss_v_raw = mse_loss((value + 1.0) / 2.0, result)
                
                if args.use_per:
                    # サンプルの損失から優先度更新
                    with torch.no_grad():
                        errors = (loss_p_raw + loss_v_raw.squeeze()).cpu().numpy()
                        sampler.update_priorities(batch_indices, errors)
                    
                    loss = loss_p_raw.mean() + loss_v_raw.mean()
                else:
                    loss = loss_p_raw + loss_v_raw
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                current_step += 1
                running_p_loss += loss_p_raw.mean().item()
                running_v_loss += loss_v_raw.mean().item()
                
                if current_step % args.eval_interval == 0 and rank == 0:
                    avg_p = running_p_loss / (step + 1)
                    avg_v = running_v_loss / (step + 1)
                    
                    # バリデーション
                    model.eval()
                    with torch.no_grad():
                        tx, tml, tres = test_loader.sample()
                        lp_t, v_t = model(tx)
                        test_pa = accuracy(lp_t, tml)
                        test_va = binary_accuracy(v_t, tres)
                    
                    logger.info("[%s] Epoch %d Step %d | Train Loss(P/V): %.4f/%.4f | Test Acc(P/V): %.4f/%.4f", 
                                mode, epoch_idx, current_step, avg_p, avg_v, test_pa, test_va)
                    
                    history[mode]["step"].append(current_step)
                    history[mode]["p_loss"].append(avg_p)
                    history[mode]["v_loss"].append(avg_v)
                    history[mode]["t_loss"].append(avg_p + avg_v)
                    save_loss_graph(history, rank=rank)
                    model.train()

            if rank == 0:
                cp_name = args.checkpoint_name.format(model=mode, epoch=epoch_idx)
                save_checkpoint(os.path.join(args.checkpoint_dir, cp_name), model, optimizer, epoch_idx, current_step, rank)
                logger.info("Epoch %d finished. Checkpoint saved: %s", epoch_idx, cp_name)

            if stop_requested:
                if rank == 0:
                    logger.info("Interrupt received. Saving state...")
                    save_checkpoint(os.path.join(args.checkpoint_dir, "interrupted.pth"), model, optimizer, epoch_idx, current_step, rank)
                break

    if world_size > 1:
        cleanup_dist()

# ===== Entry Point =====

def main():
    parser = argparse.ArgumentParser(description="Professional Imitation Learning with PER & DDP")
    parser.add_argument("--train-data", nargs="+", required=True, help="Path to training kifu files (HCPE/PSV)")
    parser.add_argument("--test-data", required=True, help="Path to test kifu file")
    parser.add_argument("--models", default="modelD:50", help="Model config and epochs (e.g. modelD:50)")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0], help="List of GPU IDs to use for DDP")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size per GPU")
    parser.add_argument("--test-batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--checkpoint-dir", default="weights_imitation/")
    parser.add_argument("--checkpoint-name", default="imitation-{model}-ep{epoch:03}.pth")
    parser.add_argument("--resume", default="", help="Checkpoint to resume from")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log", default="imitation_train.log")
    parser.add_argument("--input-features", type=int, default=0)
    parser.add_argument("--format", default="psv", choices=["hcpe", "psv"])
    parser.add_argument("--use-per", action="store_true", help="Enable Prioritized Experience Replay sampling")
    parser.add_argument("--backend", default="gloo", choices=["gloo", "nccl"], help="DDP communication backend")
    
    args = parser.parse_args()
    
    model_configs = []
    for m_spec in args.models.split(","):
        parts = m_spec.split(":")
        model_configs.append((parts[0], int(parts[1]) if len(parts) > 1 else 1))

    world_size = len(args.gpu_ids)
    
    if world_size > 1:
        import torch.multiprocessing as mp
        # マルチプロセスでの並列実行開始
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, args, model_configs))
    else:
        # 単一プロセス
        train_worker(0, 1, args, model_configs)

if __name__ == "__main__":
    main()
