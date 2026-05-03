"""
train/reinforcement_train.py  —  Gumbel AlphaZero 強化学習 (v2)
================================================================
修正点:
  [Fix-1] Gradient Clipping   : scaler.unscale_ 後に clip_grad_norm_(max=1.0)
  [Fix-2] v_targets スケール  : value head は Tanh → [-1,1]。
                                 バッファから来る v_targets が [0,1] 前提なら
                                 (v_pred+1)/2 に変換してから MSE を取る。
                                 [0,1] か [-1,1] かをフラグで明示的に制御。
  [Fix-3] LR Warmup           : linear warmup → cosine decay
  [Fix-4] load_selfplay_data  : stub を実際の npz ロードに拡充
  [Fix-5] weights/ ディレクトリ: 保存前に os.makedirs で作成
  [Fix-6] SummaryWriter クローズ: KeyboardInterrupt 時に writer.close()
  [Fix-7] Aux 損失を RL ループに組み込み (オプション)
"""

import os
import sys
import time
import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.model import create_model
from data.buffer import PrioritizedReplayBuffer, Experience
from game.board import FEATURES_SETTINGS, MOVE_LABELS_NUM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_GRAD_NORM = 1.0    # [Fix-1]
W_SCALAR      = 0.1
W_MAP         = 0.2

# [Fix-2] value head のスケール規約
#   True  → v_targets は [0,1] で来る。model出力 (Tanh→[-1,1]) を (v+1)/2 に変換して MSE
#   False → v_targets は [-1,1] で来る。変換なしで MSE
VALUE_TARGET_01 = True


# ---------------------------------------------------------------------------
# LR スケジューラ
# ---------------------------------------------------------------------------
def get_lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Aux 損失
# ---------------------------------------------------------------------------
_MSE = nn.MSELoss()
_BCE = nn.BCEWithLogitsLoss()

def compute_aux_loss(aux_pred: dict, aux_lbl: dict, device: torch.device) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device)
    loss = loss + W_SCALAR * _MSE(aux_pred["king_safety"], aux_lbl["king_safety"])
    loss = loss + W_SCALAR * _MSE(aux_pred["material"],    aux_lbl["material"])
    loss = loss + W_SCALAR * _MSE(aux_pred["mobility"],    aux_lbl["mobility"])
    loss = loss + W_MAP    * _BCE(aux_pred["attack"],       aux_lbl["attack"])
    loss = loss + W_MAP    * _BCE(aux_pred["threat"],       aux_lbl["threat"])
    loss = loss + W_MAP    * _BCE(aux_pred["damage"],       aux_lbl["damage"])
    return loss


# ---------------------------------------------------------------------------
# GumbelRLTrainer
# ---------------------------------------------------------------------------
class GumbelRLTrainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )

        self.model = create_model(
            input_channels=args.input_ch,
            num_actions=MOVE_LABELS_NUM,
            mode=args.mode,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=1e-4
        )
        use_amp      = (self.device.type == "cuda")
        self.scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        self.buffer  = PrioritizedReplayBuffer(capacity=args.capacity)

        self.writer  = SummaryWriter(log_dir=f"runs/rl_{args.mode}")  # [Fix-6]

        self.global_step  = 0
        self.warmup_steps = args.warmup_steps

        # [Fix-5] weights ディレクトリ
        os.makedirs("weights", exist_ok=True)

        # [Fix-4] ロード済みファイルを追跡
        self._loaded_files: set = set()

    # ------------------------------------------------------------------
    # [Fix-4] selfplay_data の実際のロード
    # ------------------------------------------------------------------
    def load_selfplay_data(self):
        data_dir = Path("selfplay_data")
        if not data_dir.exists():
            return

        files = sorted(data_dir.glob("*.npz"))
        new_files = [f for f in files if str(f) not in self._loaded_files]
        if not new_files:
            return

        logger.info("Loading %d new selfplay file(s)...", len(new_files))
        for fpath in new_files:
            try:
                data = np.load(fpath)
                states         = data["states"]          # (N, C, 9, 9) float32
                policy_targets = data["policy_targets"]  # (N, MOVE_LABELS_NUM) float32
                value_targets  = data["value_targets"]   # (N,) float32, range [0,1]
                # aux が含まれていれば読む
                aux_targets    = {k: data[k] for k in data.files
                                  if k not in ("states", "policy_targets", "value_targets")}

                for i in range(len(states)):
                    vt = value_targets[i]
                    # [Fix-2] バッファには [0,1] で格納
                    at = {k: aux_targets[k][i] for k in aux_targets} if aux_targets else {}
                    exp = Experience(
                        state         = states[i],
                        policy_target = policy_targets[i],
                        value_target  = np.array([vt], dtype=np.float32),
                        aux_target    = at,
                    )
                    self.buffer.add(exp, priority=1.0)

                self._loaded_files.add(str(fpath))
                logger.info("  Loaded %s (%d positions, buffer=%d)",
                            fpath.name, len(states), len(self.buffer))
            except Exception as e:
                logger.warning("Failed to load %s: %s", fpath, e)

    # ------------------------------------------------------------------
    # 学習ステップ
    # ------------------------------------------------------------------
    def train_step(self):
        if len(self.buffer) < self.args.batch:
            return None

        exps, indices, is_weights = self.buffer.sample(self.args.batch)

        states     = torch.stack([torch.from_numpy(e.state)
                                  for e in exps]).to(self.device)
        p_targets  = torch.stack([torch.from_numpy(e.policy_target)
                                  for e in exps]).to(self.device)
        # [Fix-2] value_target は [0,1] で格納されている前提
        v_targets  = torch.stack([torch.from_numpy(e.value_target)
                                  for e in exps]).to(self.device)          # (B, 1)
        is_weights_t = torch.from_numpy(is_weights).to(self.device)

        # Aux ターゲットがあれば取り出す
        has_aux    = self.args.use_aux and all(e.aux_target for e in exps)
        aux_lbl    = None
        if has_aux:
            aux_keys = list(exps[0].aux_target.keys())
            aux_lbl  = {k: torch.stack([torch.from_numpy(e.aux_target[k])
                                         for e in exps]).to(self.device)
                        for k in aux_keys}

        # [Fix-3] LR warmup
        lr_scale = get_lr_scale(self.global_step, self.warmup_steps,
                                 self.args.total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.args.lr * lr_scale

        self.model.train()
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            if has_aux:
                p_logits, v_pred, aux_pred = self.model(states, return_aux=True)
            else:
                p_logits, v_pred = self.model(states, return_aux=False)

            # Policy Loss (Gumbel AlphaZero: soft cross-entropy)
            loss_p_raw = -(p_targets * F.log_softmax(p_logits, dim=1)).sum(dim=1)

            # [Fix-2] Value Loss: model は Tanh → [-1,1]、ターゲットは [0,1]
            if VALUE_TARGET_01:
                v_pred_scaled = (v_pred + 1.0) / 2.0   # [-1,1] → [0,1]
            else:
                v_pred_scaled = v_pred                   # ターゲットが [-1,1] の場合
            loss_v_raw = F.mse_loss(v_pred_scaled, v_targets,
                                    reduction="none").squeeze(1)

            # PER の IS 重みを適用
            loss_pv = (is_weights_t * (loss_p_raw + loss_v_raw)).mean()

            # [Fix-7] Aux Loss (オプション)
            if has_aux:
                loss_aux = compute_aux_loss(aux_pred, aux_lbl, self.device)
                loss = loss_pv + loss_aux
            else:
                loss_aux = torch.tensor(0.0, device=self.device)
                loss     = loss_pv

        # [Fix-2] scaler.scale → backward
        self.scaler.scale(loss).backward()

        # [Fix-1] Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), MAX_GRAD_NORM
        )

        # NaN/Inf 検知
        if not torch.isfinite(grad_norm):
            logger.warning("Step %d: non-finite grad norm, skipping.",
                           self.global_step)
            self.optimizer.zero_grad()
            self.scaler.update()
            return None

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # PER 優先度更新
        td_errors = (loss_p_raw + loss_v_raw).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        self.global_step += 1
        return (loss.item(),
                loss_p_raw.mean().item(),
                loss_v_raw.mean().item(),
                loss_aux.item())

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------
    def run(self):
        logger.info("Starting RL loop (mode=%s, device=%s, AMP=%s)",
                    self.args.mode, self.device, self.use_amp)
        logger.info("  Warmup: %d steps | Grad clip: %.1f | v_target_01: %s",
                    self.warmup_steps, MAX_GRAD_NORM, VALUE_TARGET_01)

        try:
            while True:
                self.load_selfplay_data()

                for _ in range(self.args.steps_per_iter):
                    result = self.train_step()
                    if result is None:
                        continue
                    l, lp, lv, la = result
                    if self.global_step % 10 == 0:
                        logger.info(
                            "Step %d: Loss=%.4f (P=%.4f V=%.4f Aux=%.4f) lr_scale=%.4f",
                            self.global_step, l, lp, lv, la,
                            get_lr_scale(self.global_step,
                                         self.warmup_steps, self.args.total_steps),
                        )
                        self.writer.add_scalar("Loss/total",  l,  self.global_step)
                        self.writer.add_scalar("Loss/policy", lp, self.global_step)
                        self.writer.add_scalar("Loss/value",  lv, self.global_step)
                        self.writer.add_scalar("Loss/aux",    la, self.global_step)

                if (self.global_step > 0
                        and self.global_step % self.args.save_interval == 0):
                    ckpt_path = f"weights/rl_{self.args.mode}_step{self.global_step}.pth"
                    torch.save({
                        "step":      self.global_step,
                        "model":     self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scaler":    self.scaler.state_dict(),
                    }, ckpt_path)
                    logger.info("Saved checkpoint: %s", ckpt_path)

                time.sleep(10)

        except KeyboardInterrupt:
            logger.info("Training interrupted. Saving final state...")
            torch.save({
                "step":      self.global_step,
                "model":     self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler":    self.scaler.state_dict(),
            }, f"weights/rl_{self.args.mode}_interrupted.pth")
        finally:
            # [Fix-6] SummaryWriter を必ずクローズ
            self.writer.close()
            logger.info("SummaryWriter closed.")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gumbel AlphaZero RL Trainer (v2, 40-block safe)")
    parser.add_argument("--mode",          default="modelD")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--batch",         type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--capacity",      type=int,   default=500_000)
    parser.add_argument("--steps-per-iter",type=int,   default=100)
    parser.add_argument("--total-steps",   type=int,   default=500_000,
                        help="warmup/decay スケジューラの総ステップ数目安")
    parser.add_argument("--warmup-steps",  type=int,   default=2_000)
    parser.add_argument("--save-interval", type=int,   default=1_000)
    parser.add_argument("--input-ch",      type=int,   default=46)
    parser.add_argument("--use-aux",       action="store_true",
                        help="Aux タスク損失を RL ループに組み込む")
    args = parser.parse_args()

    trainer = GumbelRLTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
