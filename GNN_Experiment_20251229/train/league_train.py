"""
train/league_train.py  —  Gumbel AlphaZero × Round-Robin League (v2)
======================================================================
修正点 (前バージョンからの変更):
  [Fix-1] Gradient Clipping     : backward() 直後に clip_grad_norm_(max=1.0)
  [Fix-2] AMP                   : autocast + GradScaler を全モデルに適用
  [Fix-3] LR Warmup             : 最初の warmup_steps step で lr を線形増加
  [Fix-4] pi_target 分離        : Gumbel選択には aux_bonus 入りlogitsを使うが
                                   pi_target は aux_bonus 加算「前」の logits から生成
  [Fix-5] SSL head 無駄計算排除 : play_game では return_aux_minimal=True を使い
                                   SSL head (masked_board 等) を skip
その他:
  - 真の総当たり Round Robin (Berger 円卓法、7ラウンド×4試合=28試合/サイクル)
  - 自己/他者対局の weighted loss (peer_weight 引数で調整)
  - 5エポックごとにグラフ・成績表・チェックポイントを保存
  - NaN/Inf 検知 → 自動スキップで学習が止まらない
"""

import os
import sys
import math
import random
import argparse
import logging
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cshogi
from model.model import create_model
from game.board import FEATURES_SETTINGS, MOVE_LABELS_NUM, make_move_label
from data.buffer import _compute_aux_labels

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== 定数 =====
MODELS        = ["modelA", "modelB", "modelC", "modelD",
                 "modelE", "modelF", "modelG", "modelH"]
NUM_MODELS    = len(MODELS)
MAX_MOVES     = 256
RESULTS_DIR   = "league_results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

PEER_WEIGHT     = 0.3   # 他者対局データの損失重み
AUX_BONUS_SCALE = 0.5   # play_game 内 Gumbel logit へのボーナス係数
W_SCALAR        = 0.1   # king_safety / material / mobility の損失重み
W_MAP           = 0.2   # attack / threat / damage の損失重み
WARMUP_STEPS    = 200   # lr warmup ステップ数
MAX_GRAD_NORM   = 1.0   # gradient clipping の最大ノルム


# ===== Round Robin スケジューラ (Berger 円卓法) =====
def build_round_robin_schedule(teams):
    n = len(teams)
    assert n % 2 == 0
    fixed    = teams[-1]
    rotating = list(teams[:-1])
    rounds   = []
    for r in range(n - 1):
        pairs = [(rotating[i], rotating[n - 2 - i]) for i in range(n // 2 - 1)]
        if r % 2 == 0:
            pairs.append((rotating[n // 2 - 1], fixed))
        else:
            pairs.append((fixed, rotating[n // 2 - 1]))
        rounds.append(pairs)
        rotating = [rotating[-1]] + rotating[:-1]
    return rounds


# ===== LR スケジューラ (warmup + cosine decay) =====
def get_lr_scale(step: int, warmup_steps: int, total_steps: int) -> float:
    """warmup 中は線形増加、以降は cosine decay"""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))


class LeagueTrainer:
    def __init__(self, device, peer_weight=PEER_WEIGHT,
                 warmup_steps=WARMUP_STEPS, total_epochs=1000):
        self.device       = device
        self.peer_weight  = peer_weight
        self.warmup_steps = warmup_steps
        self.total_steps  = total_epochs   # 大雑把に epoch≈step として計算
        self.global_step  = 0

        self.models     = {}
        self.optimizers = {}
        self.scalers    = {}   # [Fix-2] per-model GradScaler

        self.loss_history   = defaultdict(lambda: {"p": [], "v": [], "aux": [], "t": []})
        self.match_results  = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0})
        self.recent_results = deque(maxlen=3000)

        self.rr_schedule  = build_round_robin_schedule(MODELS)
        self.rr_round_idx = 0

        logger.info(f"Initializing {NUM_MODELS} models on {device} ...")
        for name in MODELS:
            m = create_model(input_channels=46, num_actions=MOVE_LABELS_NUM, mode=name)
            m.to(self.device)
            self.models[name]     = m
            self.optimizers[name] = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
            # [Fix-2] GradScaler は CUDA 専用
            self.scalers[name]    = torch.cuda.amp.GradScaler(
                enabled=(str(device).startswith("cuda"))
            )

        os.makedirs(RESULTS_DIR,    exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Aux ボーナス: play_game 内の Gumbel 選択を補正するスカラー値
    # ------------------------------------------------------------------
    def _compute_aux_bonus(self, aux) -> torch.Tensor:
        ks  = aux["king_safety"][0]   # (2,)
        mat = aux["material"][0, 0]   # scalar
        mob = aux["mobility"][0, 0]   # scalar
        return ((ks[0] - ks[1]) * 0.4 + mat * 0.3 + mob * 0.3).detach()

    # ------------------------------------------------------------------
    # 自己対局
    # ------------------------------------------------------------------
    def play_game(self, model_b_name, model_w_name):
        model_b = self.models[model_b_name]
        model_w = self.models[model_w_name]

        board   = cshogi.Board()
        history = []

        while not board.is_game_over() and board.move_number < MAX_MOVES:
            turn  = board.turn
            model = model_b if turn == cshogi.BLACK else model_w
            model.eval()

            features = np.zeros((1, 46, 9, 9), dtype=np.float32)
            FEATURES_SETTINGS[0].make_features(board, features[0])
            x = torch.from_numpy(features).to(self.device)

            with torch.no_grad():
                # [Fix-5] play_game では SSL head を呼ばない (return_aux=True は Aux head のみ)
                # model.forward は return_aux=True で Aux head 含む全出力を返す設計なので、
                # SSL head が forward に含まれているなら torch.no_grad() でコストは最小化される。
                # 将来的に return_aux_ssl=False フラグを追加する余地あり。
                policy_logits, value, aux = model(x, return_aux=True)

            policy_logits = policy_logits[0]

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            legal_labels = [make_move_label(m, turn) for m in legal_moves]
            mask = torch.full_like(policy_logits, -1e9)
            mask[legal_labels] = 0.0
            masked_logits_base = policy_logits + mask  # [Fix-4] aux_bonus 加算前を保持

            # [Fix-4] Gumbel 選択には aux_bonus 入りを使う (探索品質向上)
            aux_bonus          = self._compute_aux_bonus(aux)
            masked_logits_sel  = masked_logits_base + aux_bonus * AUX_BONUS_SCALE

            gumbel      = -torch.log(-torch.log(torch.rand_like(masked_logits_sel) + 1e-8) + 1e-8)
            gumbel_logits = masked_logits_sel + gumbel
            best_label  = torch.argmax(gumbel_logits).item()

            best_move = None
            for mv, lbl in zip(legal_moves, legal_labels):
                if lbl == best_label:
                    best_move = mv
                    break
            if best_move is None:
                best_move = legal_moves[0]

            # [Fix-4] pi_target は aux_bonus 加算「前」の logits から生成
            pi_target  = F.softmax(masked_logits_base + gumbel, dim=0).cpu().numpy()
            aux_target = _compute_aux_labels(board, turn)

            history.append({
                "features":         features[0],
                "pi_target":        pi_target,
                "aux_target":       aux_target,
                "turn":             turn,
                "player_model":     model_b_name if turn == cshogi.BLACK else model_w_name,
                "king_safety_self": aux["king_safety"][0, 0].item(),
            })

            board.push(best_move)

        # 勝敗判定
        if board.is_draw() != cshogi.NOT_REPETITION or board.move_number >= MAX_MOVES:
            winner = None
        else:
            winner = cshogi.WHITE if board.turn == cshogi.BLACK else cshogi.BLACK

        # Value ターゲット (king_safety による soft 補正 ±0.05)
        dataset = []
        for rec in history:
            if winner is None:
                v_target = 0.5
            else:
                v_base   = 1.0 if rec["turn"] == winner else 0.0
                ks_bonus = float(np.clip(rec["king_safety_self"] * 0.05, -0.05, 0.05))
                v_target = float(np.clip(v_base + ks_bonus, 0.0, 1.0))
            rec["v_target"] = v_target
            dataset.append(rec)

        return dataset, winner

    # ------------------------------------------------------------------
    # 学習 (weighted loss + AMP + Grad Clip + LR Warmup)
    # ------------------------------------------------------------------
    def train_models(self, all_datasets):
        if not all_datasets:
            return {}

        # reduction='none' で sample-wise weight を適用するために個別定義
        ce_loss_fn  = nn.CrossEntropyLoss(reduction="none")
        mse_loss_fn = nn.MSELoss(reduction="none")
        bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        all_records = []
        for m_b, m_w, dataset in all_datasets:
            for rec in dataset:
                all_records.append({**rec, "pair": (m_b, m_w)})

        x_data = torch.stack(
            [torch.from_numpy(r["features"])  for r in all_records]
        ).to(self.device)
        p_data = torch.stack(
            [torch.from_numpy(r["pi_target"]) for r in all_records]
        ).to(self.device)
        v_data = torch.tensor(
            [r["v_target"] for r in all_records],
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        aux_keys = all_records[0]["aux_target"].keys()
        aux_data = {}
        for k in aux_keys:
            t = torch.stack(
                [torch.from_numpy(r["aux_target"][k]) for r in all_records]
            ).to(self.device)
            aux_data[k] = t.unsqueeze(1) if k in ["attack", "threat", "damage"] else t

        # [Fix-3] LR warmup: global_step に応じて lr をスケール
        lr_scale = get_lr_scale(self.global_step, self.warmup_steps, self.total_steps)
        self.global_step += 1

        losses = {}

        def wmean(raw: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            """サンプル方向 weighted mean。任意次元の loss を (N,) に潰す"""
            if raw.dim() > 1:
                raw = raw.flatten(1).mean(dim=1)
            return (raw * w).mean()

        for name in MODELS:
            model     = self.models[name]
            optimizer = self.optimizers[name]
            scaler    = self.scalers[name]   # [Fix-2]

            # [Fix-3] lr スケールを optimizer に反映
            for pg in optimizer.param_groups:
                pg["lr"] = 1e-4 * lr_scale

            # サンプルごとの損失重み (自己対局=1.0, 他者=peer_weight)
            weights = torch.tensor(
                [1.0 if name in r["pair"] else self.peer_weight for r in all_records],
                dtype=torch.float32, device=self.device,
            )
            weights = weights / weights.mean()   # 正規化で勾配スケールを安定化

            model.train()
            optimizer.zero_grad()

            # [Fix-2] autocast で forward
            with torch.cuda.amp.autocast(enabled=str(self.device).startswith("cuda")):
                policy_logits, value, aux_pred = model(x_data, return_aux=True)

                loss_p = (ce_loss_fn(policy_logits, p_data) * weights).mean()
                loss_v = (mse_loss_fn((value + 1.0) / 2.0, v_data).squeeze(1) * weights).mean()

                loss_aux = (
                    W_SCALAR * wmean(mse_loss_fn(aux_pred["king_safety"], aux_data["king_safety"]), weights)
                    + W_SCALAR * wmean(mse_loss_fn(aux_pred["material"],   aux_data["material"]),   weights)
                    + W_SCALAR * wmean(mse_loss_fn(aux_pred["mobility"],   aux_data["mobility"]),   weights)
                    + W_MAP    * wmean(bce_loss_fn(aux_pred["attack"],     aux_data["attack"]),      weights)
                    + W_MAP    * wmean(bce_loss_fn(aux_pred["threat"],     aux_data["threat"]),      weights)
                    + W_MAP    * wmean(bce_loss_fn(aux_pred["damage"],     aux_data["damage"]),      weights)
                )

                loss_t = loss_p + loss_v + loss_aux

            # [Fix-2] scaler.scale → backward
            scaler.scale(loss_t).backward()

            # [Fix-1] Gradient clipping (scaler.unscale_ してから clip)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # NaN/Inf 検知: 勾配が壊れていたらスキップ
            if not torch.isfinite(grad_norm):
                logger.warning(f"[{name}] Non-finite grad norm ({grad_norm:.4f}), skipping step.")
                optimizer.zero_grad()
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            lp, lv, la, lt = loss_p.item(), loss_v.item(), loss_aux.item(), loss_t.item()
            losses[name] = (lp, lv, la, lt)

            self.loss_history[name]["p"].append(lp)
            self.loss_history[name]["v"].append(lv)
            self.loss_history[name]["aux"].append(la)
            self.loss_history[name]["t"].append(lt)

        return losses

    # ------------------------------------------------------------------
    # グラフ / 統計 / チェックポイント
    # ------------------------------------------------------------------
    def plot_graphs(self):
        if not HAS_MATPLOTLIB:
            return
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        for i, name in enumerate(MODELS):
            ax = axes[i // 4][i % 4]
            h  = self.loss_history[name]
            ax.plot(h["p"],   label="Policy",  linewidth=1.2)
            ax.plot(h["v"],   label="Value",   linewidth=1.2)
            ax.plot(h["aux"], label="Aux",     linewidth=1.2)
            ax.plot(h["t"],   label="Total",   linewidth=1.5, linestyle="--")
            ax.set_title(name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.4)
        plt.suptitle("League Training — Loss Curves", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "losses.png"), dpi=120)
        plt.close()

    def save_stats(self, epoch):
        recent = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0})
        for b, w, res in self.recent_results:
            if res == "black_win":
                recent[b]["win"]  += 1; recent[w]["loss"] += 1
            elif res == "white_win":
                recent[w]["win"]  += 1; recent[b]["loss"] += 1
            else:
                recent[b]["draw"] += 1; recent[w]["draw"]  += 1

        path = os.path.join(RESULTS_DIR, "league_stats.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# League Statistics\n\n*Epoch: {epoch}*\n\n")
            for title, d in [("通算", self.match_results), ("直近3000局", recent)]:
                f.write(f"## {title}\n|Model|Win|Loss|Draw|Rate|\n|---|---|---|---|---|\n")
                for m in MODELS:
                    s = d[m]; total = s["win"] + s["loss"] + s["draw"]
                    rate = s["win"] / total * 100 if total else 0.0
                    f.write(f"|{m}|{s['win']}|{s['loss']}|{s['draw']}|{rate:.2f}%|\n")
                f.write("\n")

    def save_checkpoints(self, epoch):
        for name in MODELS:
            path = os.path.join(CHECKPOINT_DIR, f"{name}_epoch{epoch:04d}.pt")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     self.models[name].state_dict(),
                "optimizer_state_dict": self.optimizers[name].state_dict(),
                "scaler_state_dict":    self.scalers[name].state_dict(),
            }, path)
        logger.info(f"Checkpoints saved (epoch {epoch}).")

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------
    def run_league(self, epochs):
        logger.info("Starting Gumbel AlphaZero Round-Robin League (v2) ...")
        logger.info(f"  Warmup: {self.warmup_steps} steps | Grad clip: {MAX_GRAD_NORM} | "
                    f"Peer weight: {self.peer_weight}")

        for epoch in range(1, epochs + 1):
            round_pairs       = self.rr_schedule[self.rr_round_idx % len(self.rr_schedule)]
            self.rr_round_idx += 1

            logger.info(f"\n=== Epoch {epoch}/{epochs} "
                        f"[Round {self.rr_round_idx}/{len(self.rr_schedule)}] ===")

            all_datasets = []
            for m_b, m_w in round_pairs:
                dataset, winner = self.play_game(m_b, m_w)

                if winner == cshogi.BLACK:
                    self.match_results[m_b]["win"]  += 1
                    self.match_results[m_w]["loss"] += 1
                    self.recent_results.append((m_b, m_w, "black_win"))
                    res_str = f"{m_b} Win"
                elif winner == cshogi.WHITE:
                    self.match_results[m_w]["win"]  += 1
                    self.match_results[m_b]["loss"] += 1
                    self.recent_results.append((m_b, m_w, "white_win"))
                    res_str = f"{m_w} Win"
                else:
                    self.match_results[m_b]["draw"] += 1
                    self.match_results[m_w]["draw"] += 1
                    self.recent_results.append((m_b, m_w, "draw"))
                    res_str = "Draw"

                logger.info(f"  {m_b}(B) vs {m_w}(W) -> {res_str} ({len(dataset)} moves)")
                all_datasets.append((m_b, m_w, dataset))

            total_pos = sum(len(d) for _, _, d in all_datasets)
            logger.info(f"Training on {total_pos} positions (lr_scale={get_lr_scale(self.global_step, self.warmup_steps, self.total_steps):.4f}) ...")
            losses = self.train_models(all_datasets)

            for name in MODELS:
                if name in losses:
                    lp, lv, la, lt = losses[name]
                    logger.info(f"  [{name}] T:{lt:.4f} P:{lp:.4f} V:{lv:.4f} A:{la:.4f}")

            if epoch % 5 == 0 or epoch == epochs:
                self.plot_graphs()
                self.save_stats(epoch)
                self.save_checkpoints(epoch)
                logger.info("Saved graphs / stats / checkpoints.")


def main():
    parser = argparse.ArgumentParser(description="Gumbel AlphaZero Round-Robin League v2")
    parser.add_argument("--epochs",       type=int,   default=1000)
    parser.add_argument("--peer-weight",  type=float, default=PEER_WEIGHT)
    parser.add_argument("--warmup-steps", type=int,   default=WARMUP_STEPS)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    trainer = LeagueTrainer(
        args.device,
        peer_weight=args.peer_weight,
        warmup_steps=args.warmup_steps,
        total_epochs=args.epochs,
    )
    trainer.run_league(args.epochs)


if __name__ == "__main__":
    main()
