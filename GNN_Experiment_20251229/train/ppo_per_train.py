"""
train.py  —  HybridAlphaZeroNet 強化学習トレーニングループ
==============================================================

機能
----
* 混合精度学習 (AMP / torch.cuda.amp)
* PPO (Proximate Policy Optimization) による方策更新
* PER (Prioritized Experience Replay) による優先度付きサンプリング
* マルチタスク損失
    - L_policy  : PPO clipped surrogate
    - L_value   : MSE (value head)
    - L_entropy : エントロピーボーナス（方策の多様性維持）
    - L_aux     : 補助タスク群 (king_safety / material / mobility /
                                attack / threat / damage)
    - L_ssl     : 自己教師あり (masked_board 再構成)
* 対局時は model(x, return_aux=False) で policy + value のみ使用
* チェックポイント保存 / 再開 (weights/ ディレクトリ)
* TensorBoard ログ出力（オプションで wandb にも対応）

使い方
------
  python train.py --mode modelD --device cuda --batch_size 512

ディレクトリ構成（プロジェクトルートから）
  data/buffer.py      ← このファイルの buffer.py
  model/model.py      ← HybridAlphaZeroNet
  search/mcts.py      ← MCTS（self_play_one_game で呼ぶ）
  game/board.py       ← 将棋ルール・補助ターゲット計算
  train/train.py      ← このファイル
  weights/            ← チェックポイント保存先
"""

import os
import sys
import math
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.model import HybridAlphaZeroNet, create_model
from data.buffer import PrioritizedReplayBuffer, Experience, collate_experiences

# ---------------------------------------------------------------------------
# ロガー設定
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# ハイパーパラメータ
# ===========================================================================
class HyperParams:
    # --- モデル ---
    INPUT_CHANNELS: int   = 46         # 盤面特徴のチャンネル数（要調整）
    NUM_ACTIONS:    int   = 13527      # 将棋の合法手上限（dlshogi 準拠）

    # --- 学習全般 ---
    TOTAL_GAMES:    int   = 100_000    # 自己対局ゲーム数（総計）
    BATCH_SIZE:     int   = 512
    LEARNING_RATE:  float = 2e-4
    WEIGHT_DECAY:   float = 1e-4
    GRAD_CLIP:      float = 1.0        # 勾配クリッピング閾値

    # --- PPO ---
    PPO_EPOCHS:     int   = 4          # 1 バッチのデータで何回更新するか
    PPO_CLIP:       float = 0.2        # ε：policy ratio のクリップ幅
    VALUE_COEF:     float = 0.5        # value loss の係数
    ENTROPY_COEF:   float = 0.01       # entropy bonus の係数（探索促進）

    # --- マルチタスク損失係数 ---
    AUX_COEF:       float = 0.1        # 補助タスク全体の係数
    SSL_COEF:       float = 0.05       # 自己教師あり損失の係数

    # --- PER ---
    BUFFER_CAPACITY: int  = 500_000
    PER_ALPHA:       float = 0.6
    PER_BETA_START:  float = 0.4
    PER_BETA_FRAMES: int   = 1_000_000
    MIN_BUFFER_SIZE: int   = 10_000    # 学習開始に必要な最低経験数

    # --- 自己対局 ---
    SELF_PLAY_GAMES_PER_ITER: int = 10  # 1 学習イテレーションあたりの対局数
    MCTS_SIMULATIONS:         int = 200
    TEMPERATURE_THRESHOLD:    int = 30  # この手数以降は temperature=0（最善手選択）
    TEMPERATURE_HIGH:         float = 1.0
    TEMPERATURE_LOW:          float = 0.1

    # --- SSL マスキング ---
    MASK_RATIO: float = 0.15           # 盤面の何割をマスクするか

    # --- 学習率スケジューラ ---
    LR_WARMUP_STEPS: int = 1_000       # ウォームアップステップ数

    # --- ログ・チェックポイント ---
    LOG_INTERVAL:       int = 100      # 何ステップごとにログを出すか
    CHECKPOINT_INTERVAL:int = 1_000   # 何ステップごとにチェックポイントを保存するか
    WEIGHTS_DIR:        str = "weights"


HP = HyperParams()


# ===========================================================================
# 損失関数群
# ===========================================================================

def ppo_policy_loss(
    new_log_probs: torch.Tensor,   # (B,)  現在の方策の log π(a|s)
    old_log_probs: torch.Tensor,   # (B,)  自己対局時の log π(a|s)
    advantages:    torch.Tensor,   # (B,)  アドバンテージ推定値
    clip:          float = HP.PPO_CLIP,
) -> torch.Tensor:
    """
    PPO のクリップ付き代理目的関数（最大化 → 負にして最小化）。

    L_CLIP = -E[ min(r * A, clip(r, 1-ε, 1+ε) * A) ]
    """
    ratio = (new_log_probs - old_log_probs).exp()   # π_new / π_old
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    return -torch.min(surr1, surr2)   # shape (B,)  ← IS 重みを後で掛ける


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    方策エントロピー H(π) を計算して返す（大きいほど多様な探索）。
    損失として使う場合は符号を反転して -H を最小化する。
    ここでは +H を返し、呼び出し元で -ENTROPY_COEF * H を足す。
    """
    probs     = F.softmax(logits, dim=1)       # (B, num_actions)
    log_probs = F.log_softmax(logits, dim=1)   # (B, num_actions)
    return -(probs * log_probs).sum(dim=1)     # (B,)


def value_loss(
    pred:   torch.Tensor,   # (B, 1)
    target: torch.Tensor,   # (B, 1)
) -> torch.Tensor:
    """Huber Loss（MSE より外れ値に頑健）。shape (B,)"""
    return F.huber_loss(pred, target, reduction='none').squeeze(1)


def auxiliary_loss(
    aux:    Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    補助タスク群の損失を合算して返す。shape (B,)

    タスク一覧:
      king_safety  : MSE  (B, 2)
      material     : MSE  (B, 1)
      mobility     : MSE  (B, 1)
      attack_map   : BCE  (B, 1, 9, 9)
      threat_map   : BCE  (B, 1, 9, 9)
      damage_map   : BCE  (B, 1, 9, 9)
    """
    B = aux['king_safety'].shape[0]

    # スカラー系: reduction='none' で (B,) 形状に揃える
    l_king     = F.mse_loss(aux['king_safety'],
                            targets['king_safety'],
                            reduction='none').mean(dim=1)            # (B,)
    l_material = F.mse_loss(aux['material'],
                            targets['material'],
                            reduction='none').squeeze(1)             # (B,)
    l_mobility = F.mse_loss(aux['mobility'],
                            targets['mobility'],
                            reduction='none').squeeze(1)             # (B,)

    # マップ系: (B, 1, 9, 9) → 空間方向で平均 → (B,)
    l_attack = F.binary_cross_entropy(
        aux['attack'], targets['attack_map'], reduction='none'
    ).mean(dim=(1, 2, 3))
    l_threat = F.binary_cross_entropy(
        aux['threat'], targets['threat_map'], reduction='none'
    ).mean(dim=(1, 2, 3))
    l_damage = F.binary_cross_entropy(
        aux['damage'], targets['damage_map'], reduction='none'
    ).mean(dim=(1, 2, 3))

    return (l_king + l_material + l_mobility +
            l_attack + l_threat + l_damage) / 6.0   # (B,)


def ssl_loss(
    aux:    Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    自己教師あり損失: masked_board の駒種クラス分類。

    aux['masked_board']  : (B, num_piece_types, 9, 9) — logits
    targets['masked_board']: (B, 9, 9)                — class indices
    targets['mask_indices'] : (B, K)                  — マスクした升目

    マスクした升目のみを対象に cross_entropy を計算する。
    """
    B = aux['masked_board'].shape[0]
    # (B, num_piece_types, 9, 9) → (B, num_piece_types, 81)
    logits   = aux['masked_board'].view(B, -1, 81)
    t_board  = targets['masked_board'].view(B, 81)     # (B, 81)
    mask_idx = targets['mask_indices']                 # (B, K)
    K        = mask_idx.shape[1]

    # マスク升目だけ抽出: (B*K,) の CE を計算
    # gather で (B, K) インデックスに対応するロジットを取り出す
    idx_exp  = mask_idx.unsqueeze(1).expand(B, logits.shape[1], K)  # (B, C, K)
    logits_m = logits.gather(2, idx_exp).permute(0, 2, 1).reshape(B * K, -1)  # (B*K, C)
    target_m = t_board.gather(1, mask_idx).reshape(B * K)            # (B*K,)

    ce = F.cross_entropy(logits_m, target_m, reduction='none')       # (B*K,)
    return ce.view(B, K).mean(dim=1)                                  # (B,)


def compute_total_loss(
    policy_logits:  torch.Tensor,           # (B, num_actions)
    value_pred:     torch.Tensor,           # (B, 1)
    aux:            Dict[str, torch.Tensor],
    targets:        Dict[str, torch.Tensor],
    is_weights:     torch.Tensor,           # (B,)  PER IS 重み
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    全損失を計算して (total_loss scalar, 各損失の dict) を返す。

    total = IS_w * (L_policy + V_coef*L_value - E_coef*L_entropy
                    + AUX_coef*L_aux + SSL_coef*L_ssl)
    """
    # ---- policy ターゲット ----
    policy_target = targets['policy_target']           # (B, num_actions) MCTS 確率
    old_log_prob  = targets['old_log_prob'].squeeze(1) # (B,)

    # actions = argmax of MCTS policy（代表行動として TD 誤差計算に使う）
    actions       = policy_target.argmax(dim=1)        # (B,)
    log_probs_new = F.log_softmax(policy_logits, dim=1)
    new_log_prob  = log_probs_new.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    # アドバンテージ = value target - value pred（detach して勾配を切る）
    advantages = (targets['value_target'].squeeze(1)
                  - value_pred.squeeze(1).detach())    # (B,)
    # 正規化（安定化）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ---- 各損失 (shape: B,) ----
    l_policy  = ppo_policy_loss(new_log_prob, old_log_prob, advantages)
    l_value   = value_loss(value_pred, targets['value_target'])
    l_entropy = entropy_loss(policy_logits)   # H(π), 最大化したいので後で引く
    l_aux     = auxiliary_loss(aux, targets)
    l_ssl     = ssl_loss(aux, targets)

    # ---- 重み付き合算 ----
    per_sample = (l_policy
                  + HP.VALUE_COEF   * l_value
                  - HP.ENTROPY_COEF * l_entropy
                  + HP.AUX_COEF     * l_aux
                  + HP.SSL_COEF     * l_ssl)

    # IS 重みを掛けて平均（PER バイアス補正）
    total = (is_weights * per_sample).mean()

    # ---- ログ用辞書 ----
    stats = {
        'loss/total':   total.item(),
        'loss/policy':  l_policy.mean().item(),
        'loss/value':   l_value.mean().item(),
        'loss/entropy': l_entropy.mean().item(),
        'loss/aux':     l_aux.mean().item(),
        'loss/ssl':     l_ssl.mean().item(),
        'misc/advantages_mean': advantages.mean().item(),
        'misc/advantages_std':  advantages.std().item(),
    }

    return total, stats, new_log_prob.detach(), advantages.detach()


# ===========================================================================
# 自己対局 (self-play) ユーティリティ
# ===========================================================================

def generate_masked_board(
    board_state: np.ndarray,   # int64 (9, 9) 駒種クラスインデックス
    mask_ratio:  float = HP.MASK_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    盤面の一部をランダムにマスクし、
    (masked_board, mask_indices) を返す。
    masked_board: int64 (9, 9) — マスク升目は 0（空マスとして扱う）
    mask_indices: int64 (K,)   — マスクした升目のフラット インデックス
    """
    flat     = board_state.flatten()      # (81,)
    n_mask   = max(1, int(81 * mask_ratio))
    mask_idx = np.random.choice(81, n_mask, replace=False).astype(np.int64)
    masked   = flat.copy()
    masked[mask_idx] = 0                  # 0 = 空マスクラス
    return masked.reshape(9, 9), mask_idx


def self_play_one_game(
    model:  HybridAlphaZeroNet,
    device: torch.device,
    game_idx: int,
) -> List[Experience]:
    """
    モデルを使って自己対局を 1 ゲーム行い、
    棋譜から Experience のリストを生成して返す。

    NOTE: board.py / mcts.py の実装に依存する。
          ここではインターフェースを示す骨格実装。
          実際の盤面処理・補助ターゲット計算は board.py 側に実装すること。
    """
    try:
        from game.board import Board
        from search.mcts import MCTS
    except ImportError:
        logger.warning("game.board / search.mcts が見つかりません。ダミーデータで代替します。")
        return _dummy_experiences(n=30)

    model.eval()
    board = Board()
    mcts  = MCTS(model, device, n_simulations=HP.MCTS_SIMULATIONS)

    trajectory = []   # [(state_feat, policy_probs, old_log_prob, board_snapshot), ...]
    move_count = 0

    while not board.is_terminal():
        # ---- temperature の設定 ----
        temp = (HP.TEMPERATURE_HIGH
                if move_count < HP.TEMPERATURE_THRESHOLD
                else HP.TEMPERATURE_LOW)

        # ---- MCTS で指し手確率を取得 ----
        # mcts.search() は (policy_probs: np.ndarray (num_actions,),
        #                    old_log_prob: float) を返すと仮定
        policy_probs, old_log_prob = mcts.search(board, temperature=temp)

        # ---- 指し手選択 ----
        if temp == HP.TEMPERATURE_LOW:
            action = int(np.argmax(policy_probs))
        else:
            action = int(np.random.choice(len(policy_probs), p=policy_probs))

        # ---- 盤面特徴量・補助ターゲットを取得 ----
        # board.get_features() → np.float32 (C, 9, 9)
        # board.get_aux_targets() → dict of np.ndarray
        state_feat  = board.get_features()
        aux_targets = board.get_aux_targets()

        trajectory.append((
            state_feat,
            policy_probs,
            np.array([old_log_prob], dtype=np.float32),
            aux_targets,
        ))

        board.push(action)
        move_count += 1

    # ---- 終局処理: 報酬を後ろから割り当て ----
    # board.result() は先手視点での終局報酬 ∈ {1, -1, 0} を返すと仮定
    result = board.result()   # 1: 先手勝, -1: 後手勝, 0: 引き分け

    experiences = []
    for i, (state_feat, policy_probs, old_log_prob, aux_t) in enumerate(trajectory):
        # 手番交互なので偶数手番(先手)は +result、奇数手番(後手)は -result
        z = float(result) if (i % 2 == 0) else float(-result)

        # SSL 用マスキング
        # board_snapshot として aux_t 内の 'board_class' (int64, 9x9) を使う
        board_class = aux_t.get('board_class',
                                np.zeros((9, 9), dtype=np.int64))
        masked_board, mask_idx = generate_masked_board(board_class)

        exp = Experience(
            state          = state_feat,
            policy_target  = policy_probs.astype(np.float32),
            value_target   = np.array([z], dtype=np.float32),
            old_log_prob   = old_log_prob,
            king_safety    = aux_t.get('king_safety',
                                       np.zeros(2, dtype=np.float32)),
            material       = aux_t.get('material',
                                       np.zeros(1, dtype=np.float32)),
            mobility       = aux_t.get('mobility',
                                       np.zeros(1, dtype=np.float32)),
            attack_map     = aux_t.get('attack_map',
                                       np.zeros((1, 9, 9), dtype=np.float32)),
            threat_map     = aux_t.get('threat_map',
                                       np.zeros((1, 9, 9), dtype=np.float32)),
            damage_map     = aux_t.get('damage_map',
                                       np.zeros((1, 9, 9), dtype=np.float32)),
            masked_board   = masked_board,
            mask_indices   = mask_idx,
        )
        experiences.append(exp)

    return experiences


def _dummy_experiences(n: int = 30) -> List[Experience]:
    """board.py が未実装の場合にテスト用のダミーデータを返す。"""
    exps = []
    for _ in range(n):
        masked_board_cls = np.random.randint(0, 31, (9, 9), dtype=np.int64)
        masked_b, mask_idx = generate_masked_board(masked_board_cls)
        exps.append(Experience(
            state          = np.random.randn(HP.INPUT_CHANNELS, 9, 9).astype(np.float32),
            policy_target  = (lambda x: x / x.sum())(
                                np.random.rand(HP.NUM_ACTIONS).astype(np.float32)),
            value_target   = np.array([random.choice([-1., 0., 1.])],
                                      dtype=np.float32),
            old_log_prob   = np.array([np.log(1.0 / HP.NUM_ACTIONS)],
                                      dtype=np.float32),
            king_safety    = np.random.rand(2).astype(np.float32),
            material       = np.random.rand(1).astype(np.float32),
            mobility       = np.random.rand(1).astype(np.float32),
            attack_map     = np.random.rand(1, 9, 9).astype(np.float32),
            threat_map     = np.random.rand(1, 9, 9).astype(np.float32),
            damage_map     = np.random.rand(1, 9, 9).astype(np.float32),
            masked_board   = masked_b,
            mask_indices   = mask_idx,
        ))
    return exps


# ===========================================================================
# 学習率スケジューラ（線形ウォームアップ + コサイン減衰）
# ===========================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = HP.LR_WARMUP_STEPS,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===========================================================================
# チェックポイント保存 / ロード
# ===========================================================================

def save_checkpoint(
    path:          str,
    model:         nn.Module,
    optimizer:     torch.optim.Optimizer,
    scheduler:     torch.optim.lr_scheduler._LRScheduler,
    scaler:        torch.cuda.amp.GradScaler,
    global_step:   int,
    game_count:    int,
    best_loss:     float,
):
    torch.save({
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state':    scaler.state_dict(),
        'global_step':     global_step,
        'game_count':      game_count,
        'best_loss':       best_loss,
    }, path)
    logger.info(f"Checkpoint saved → {path}  (step={global_step})")


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler:    torch.cuda.amp.GradScaler,
    device:    torch.device,
) -> Tuple[int, int, float]:
    """
    チェックポイントをロードして (global_step, game_count, best_loss) を返す。
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    scaler.load_state_dict(ckpt['scaler_state'])
    logger.info(f"Checkpoint loaded ← {path}  (step={ckpt['global_step']})")
    return ckpt['global_step'], ckpt['game_count'], ckpt['best_loss']


# ===========================================================================
# PPO 更新ループ（1 バッチに対して複数エポック更新）
# ===========================================================================

def ppo_update(
    model:      HybridAlphaZeroNet,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler._LRScheduler,
    scaler:     torch.cuda.amp.GradScaler,
    buffer:     PrioritizedReplayBuffer,
    device:     torch.device,
    global_step: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    PER からバッチをサンプリングして PPO_EPOCHS 回モデルを更新する。

    Returns
    -------
    avg_stats    : Dict[str, float]  平均ログ統計
    tree_indices : np.ndarray        更新対象の SumTree インデックス
    td_errors    : np.ndarray        新しい TD 誤差（PER 優先度更新用）
    """
    experiences, tree_indices, is_weights_np = buffer.sample(HP.BATCH_SIZE)

    # numpy → tensor
    batch      = collate_experiences(experiences, device)
    is_weights = torch.tensor(is_weights_np, dtype=torch.float32, device=device)

    accumulated_stats: Dict[str, List[float]] = {}
    td_errors_list: List[np.ndarray] = []

    model.train()

    for epoch in range(HP.PPO_EPOCHS):
        with torch.cuda.amp.autocast():
            # forward
            policy_logits, value_pred, aux = model(batch['state'], return_aux=True)

            # 損失計算
            total_loss, stats, new_log_prob, advantages = compute_total_loss(
                policy_logits, value_pred, aux, batch, is_weights
            )

        # backward（AMP scaler 経由）
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), HP.GRAD_CLIP
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ログ統計の蓄積
        stats['misc/grad_norm'] = grad_norm.item()
        stats['misc/lr']        = scheduler.get_last_lr()[0]
        for k, v in stats.items():
            accumulated_stats.setdefault(k, []).append(v)

        # TD 誤差の計算: |advantage| を使う（最終エポックのみ記録）
        if epoch == HP.PPO_EPOCHS - 1:
            with torch.no_grad():
                td_err = advantages.abs().cpu().numpy()
                td_errors_list.append(td_err)

    # 平均統計
    avg_stats = {k: float(np.mean(v)) for k, v in accumulated_stats.items()}

    td_errors_final = td_errors_list[-1] if td_errors_list else np.ones(HP.BATCH_SIZE)

    return avg_stats, tree_indices, td_errors_final


# ===========================================================================
# メインのトレーニングループ
# ===========================================================================

def train(args):
    # --- デバイス設定 ---
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # --- 再現性のためのシード固定 ---
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # 固定サイズ入力で高速化

    # --- モデル構築 ---
    model = create_model(
        input_channels=HP.INPUT_CHANNELS,
        num_actions=HP.NUM_ACTIONS,
        mode=args.mode,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.mode} | Parameters: {param_count:,}")

    # --- オプティマイザ ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HP.LEARNING_RATE,
        weight_decay=HP.WEIGHT_DECAY,
        eps=1e-8,
    )

    # --- AMP GradScaler ---
    # enabled=False にすると AMP なしのフルFP32 学習になる（デバッグ用）
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # --- 推定総ステップ数（スケジューラ用） ---
    # 1 ゲームあたり平均 80 手、学習開始後は 10 ゲームに 1 回更新
    est_total_steps = (HP.TOTAL_GAMES // HP.SELF_PLAY_GAMES_PER_ITER) * HP.PPO_EPOCHS
    scheduler = build_scheduler(optimizer, total_steps=est_total_steps)

    # --- PER バッファ ---
    buffer = PrioritizedReplayBuffer(
        capacity    = HP.BUFFER_CAPACITY,
        alpha       = HP.PER_ALPHA,
        beta_start  = HP.PER_BETA_START,
        beta_frames = HP.PER_BETA_FRAMES,
    )

    # --- TensorBoard ---
    log_dir = Path("runs") / f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    writer  = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard log dir: {log_dir}")

    # --- weights ディレクトリ ---
    weights_dir = Path(HP.WEIGHTS_DIR)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # --- チェックポイント再開 ---
    global_step = 0
    game_count  = 0
    best_loss   = float('inf')

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            global_step, game_count, best_loss = load_checkpoint(
                str(resume_path), model, optimizer, scheduler, scaler, device
            )
        else:
            logger.warning(f"Resume path not found: {resume_path}")

    # =========================================================================
    # メインループ
    # =========================================================================
    logger.info("=== Training started ===")
    step_stats_buffer: List[Dict[str, float]] = []

    while game_count < HP.TOTAL_GAMES:

        # ------------------------------------------------------------------
        # Phase 1: 自己対局でバッファに経験を蓄積
        # ------------------------------------------------------------------
        t_play = time.time()
        for _ in range(HP.SELF_PLAY_GAMES_PER_ITER):
            experiences = self_play_one_game(model, device, game_count)
            # 新しい経験には最大優先度を与える
            max_priority = float(buffer._tree._tree[1:].max()) if len(buffer) > 0 else 1.0
            buffer.add_batch(
                experiences,
                td_errors=np.full(len(experiences), max_priority, dtype=np.float32),
            )
            game_count += 1
        t_play_elapsed = time.time() - t_play

        # ------------------------------------------------------------------
        # Phase 2: バッファが溜まったら学習
        # ------------------------------------------------------------------
        if not buffer.is_ready(HP.MIN_BUFFER_SIZE):
            logger.info(
                f"[Warmup] Buffer: {len(buffer):,} / {HP.MIN_BUFFER_SIZE:,} "
                f"  (games: {game_count})"
            )
            continue

        t_train = time.time()
        avg_stats, tree_indices, td_errors = ppo_update(
            model, optimizer, scheduler, scaler, buffer, device, global_step
        )
        t_train_elapsed = time.time() - t_train

        # ------------------------------------------------------------------
        # PER 優先度の更新
        # ------------------------------------------------------------------
        buffer.update_priorities(tree_indices, td_errors)

        global_step += 1

        # ------------------------------------------------------------------
        # ログ出力
        # ------------------------------------------------------------------
        avg_stats['misc/buffer_size']    = float(len(buffer))
        avg_stats['misc/game_count']     = float(game_count)
        avg_stats['misc/play_time_sec']  = t_play_elapsed
        avg_stats['misc/train_time_sec'] = t_train_elapsed

        step_stats_buffer.append(avg_stats)

        if global_step % HP.LOG_INTERVAL == 0:
            # バッファの平均を TensorBoard に書き出す
            agg: Dict[str, float] = {}
            for k in step_stats_buffer[0]:
                agg[k] = float(np.mean([s[k] for s in step_stats_buffer]))
            for k, v in agg.items():
                writer.add_scalar(k, v, global_step)
            step_stats_buffer.clear()

            logger.info(
                f"Step {global_step:6d} | Games {game_count:6d} | "
                f"Loss {agg['loss/total']:.4f} | "
                f"Policy {agg['loss/policy']:.4f} | "
                f"Value {agg['loss/value']:.4f} | "
                f"Entropy {agg['loss/entropy']:.4f} | "
                f"LR {agg['misc/lr']:.2e} | "
                f"Buffer {int(agg['misc/buffer_size']):,}"
            )

        # ------------------------------------------------------------------
        # チェックポイント保存
        # ------------------------------------------------------------------
        if global_step % HP.CHECKPOINT_INTERVAL == 0:
            # latest
            latest_path = weights_dir / f"checkpoint-{args.mode}-latest.pth"
            save_checkpoint(
                str(latest_path), model, optimizer, scheduler, scaler,
                global_step, game_count, best_loss
            )

            # best（損失が改善したとき）
            current_loss = avg_stats['loss/total']
            if current_loss < best_loss:
                best_loss   = current_loss
                best_path   = weights_dir / f"checkpoint-{args.mode}-best.pth"
                save_checkpoint(
                    str(best_path), model, optimizer, scheduler, scaler,
                    global_step, game_count, best_loss
                )
                logger.info(f"Best model updated! loss={best_loss:.4f}")

    # =========================================================================
    # 学習終了
    # =========================================================================
    final_path = weights_dir / f"checkpoint-{args.mode}-final.pth"
    save_checkpoint(
        str(final_path), model, optimizer, scheduler, scaler,
        global_step, game_count, best_loss
    )
    writer.close()
    logger.info("=== Training finished ===")


# ===========================================================================
# 対局用 inference ヘルパー（train.py から export して main.py で使う）
# ===========================================================================

@torch.no_grad()
def infer_policy_value(
    model:  HybridAlphaZeroNet,
    state:  np.ndarray,          # float32 (C, 9, 9)
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    """
    対局時の推論。policy 確率分布と value スカラーを返す。
    return_aux=False なので補助タスクは計算されない。
    """
    model.eval()
    x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)   # (1, C, 9, 9)
    log_policy, value = model(x, return_aux=False)
    policy = log_policy.exp().squeeze(0).cpu().numpy()   # (num_actions,)
    v      = value.squeeze().item()
    return policy, v


# ===========================================================================
# エントリポイント
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HybridAlphaZeroNet Trainer")
    parser.add_argument('--mode',       type=str,   default='modelD',
                        choices=['modelA', 'modelB', 'modelC', 'modelD', '30blocks'],
                        help='モデル構成（model.py の create_model に対応）')
    parser.add_argument('--device',     type=str,   default='cuda',
                        help='cuda / cpu')
    parser.add_argument('--batch_size', type=int,   default=HP.BATCH_SIZE)
    parser.add_argument('--lr',         type=float, default=HP.LEARNING_RATE)
    parser.add_argument('--resume',     type=str,   default=None,
                        help='再開するチェックポイントのパス')
    parser.add_argument('--seed',       type=int,   default=42)
    # ハイパーパラメータの上書き（実験用）
    parser.add_argument('--total_games',    type=int,   default=HP.TOTAL_GAMES)
    parser.add_argument('--ppo_clip',       type=float, default=HP.PPO_CLIP)
    parser.add_argument('--entropy_coef',   type=float, default=HP.ENTROPY_COEF)
    parser.add_argument('--aux_coef',       type=float, default=HP.AUX_COEF)
    parser.add_argument('--ssl_coef',       type=float, default=HP.SSL_COEF)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # コマンドライン引数で HP を上書き
    HP.BATCH_SIZE    = args.batch_size
    HP.LEARNING_RATE = args.lr
    HP.TOTAL_GAMES   = args.total_games
    HP.PPO_CLIP      = args.ppo_clip
    HP.ENTROPY_COEF  = args.entropy_coef
    HP.AUX_COEF      = args.aux_coef
    HP.SSL_COEF      = args.ssl_coef

    train(args)
