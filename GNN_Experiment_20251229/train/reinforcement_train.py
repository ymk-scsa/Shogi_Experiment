"""
reinforcement_train.py — Gumbel AlphaZero 式自己対局強化学習
=============================================================

このスクリプトは、自己対局データを selfplay_data/ から読み込み、
Gumbel AlphaZero 方式（方策改善と価値評価の反復）でモデルを強化します。
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.model import create_model
from data.buffer import PrioritizedReplayBuffer, Experience, ShogiDataLoader
from game.board import FEATURES_SETTINGS, MOVE_LABELS_NUM

# ===== 設定 =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class GumbelRLTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # モデル構築
        self.model = create_model(
            input_channels=args.input_ch,
            num_actions=MOVE_LABELS_NUM,
            mode=args.mode
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))
        
        # バッファ (自己対局データ用)
        self.buffer = PrioritizedReplayBuffer(capacity=args.capacity)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=f"runs/rl_{args.mode}")
        
        self.global_step = 0
        
    def load_selfplay_data(self):
        """selfplay_data フォルダからデータをロードする"""
        data_dir = Path("selfplay_data")
        files = list(data_dir.glob("*.npz")) + list(data_dir.glob("*.bin"))
        if not files:
            logger.warning("No selfplay data found in selfplay_data/")
            return
        
        logger.info(f"Loading {len(files)} selfplay data files...")
        # ここでデータをバッファに追加するロジックを実装（簡略化）
        # 実際には selfplay 生成側が buffer.save() したものを load するか、1件ずつ add する
        
    def train_step(self):
        if len(self.buffer) < self.args.batch:
            return None
        
        exps, indices, is_weights = self.buffer.sample(self.args.batch)
        # collate_experiences の代わり（簡略化）
        states = torch.stack([torch.from_numpy(e.state) for e in exps]).to(self.device)
        p_targets = torch.stack([torch.from_numpy(e.policy_target) for e in exps]).to(self.device)
        v_targets = torch.stack([torch.from_numpy(e.value_target) for e in exps]).to(self.device)
        is_weights = torch.from_numpy(is_weights).to(self.device)
        
        self.model.train()
        with torch.cuda.amp.autocast():
            p_logits, v_pred = self.model(states, return_aux=False)
            
            # Policy Loss: Cross Entropy with MCTS target
            # Gumbel AlphaZero では Improved Policy をターゲットにする
            loss_p = -(p_targets * F.log_softmax(p_logits, dim=1)).sum(dim=1)
            
            # Value Loss: MSE
            loss_v = F.mse_loss(v_pred, v_targets, reduction='none').squeeze(1)
            
            # Weighted Loss
            loss = (is_weights * (loss_p + loss_v)).mean()
            
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 優先度更新
        td_errors = (loss_p + loss_v).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)
        
        self.global_step += 1
        return loss.item(), loss_p.mean().item(), loss_v.mean().item()

    def run(self):
        logger.info("Starting Reinforcement Learning loop...")
        try:
            while True:
                # 1. データの読み込み
                self.load_selfplay_data()
                
                # 2. 学習
                for _ in range(self.args.steps_per_iter):
                    result = self.train_step()
                    if result:
                        l, lp, lv = result
                        if self.global_step % 10 == 0:
                            logger.info(f"Step {self.global_step}: Loss={l:.4f} (P={lp:.4f}, V={lv:.4f})")
                            self.writer.add_scalar("Loss/total", l, self.global_step)
                
                # 3. 保存
                if self.global_step % self.args.save_interval == 0:
                    torch.save(self.model.state_dict(), f"weights/rl_{self.args.mode}_latest.pth")
                
                time.sleep(10) # データの更新待ち
        except KeyboardInterrupt:
            logger.info("Training interrupted.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="modelD")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--capacity", type=int, default=500000)
    parser.add_argument("--steps_per_iter", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--input_ch", type=int, default=46)
    args = parser.parse_args()
    
    trainer = GumbelRLTrainer(args)
    trainer.run()

if __name__ == "__main__":
    main()
