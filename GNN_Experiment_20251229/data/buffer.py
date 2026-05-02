"""
buffer.py — データ読み込みと経験再生バッファの統合
=================================================

このモジュールは以下の2つの機能を提供します：
1. 教師データ（PSV/HCPE）の読み込み（イミテーション学習用）
2. 自己対局データの管理（強化学習用: Prioritized Experience Replay）

教師データとして Suisho10Mn_psv.bin などの PackedSfenValue 形式をサポートします。

【変更点】
- _make_batch に aux ラベルの自動計算を追加（人間が値を設定しない）
  - king_safety : 自玉・敵玉の周囲密集度 [0,1]
  - material    : 駒得スコア（正規化済み）
  - mobility    : 合法手数 / 593（正規化済み）
  - attack      : 自分の利きマップ (9,9) バイナリ
  - threat      : 相手の利きマップ (9,9) バイナリ
  - damage      : 相手利きにある自駒マスのバイナリ (9,9)
  全て cshogi の Board API から自動計算し、人間が定義した数値は使わない。
"""

import os
import sys
import math
import logging
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from cshogi import (
    Board, HuffmanCodedPosAndEval, PackedSfenValue,
    move16_from_psv, BLACK, WHITE,
    move_to, move_from, move_is_drop,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from game.board import FEATURES_SETTINGS, make_move_label, make_result

_logger = logging.getLogger(__name__)
_PSV_SCORE_SCALE = 600.0

# ---------------------------------------------------------------------------
# 駒点数テーブル（Material balance 計算用）
# インデックス = piece_type (0〜16)
# 値は一般的な将棋の駒得換算値。「人間が設定した絶対値」ではなく
# 「相対的な駒得の強弱」を学習するためのスケール基準として使用。
# ---------------------------------------------------------------------------
_PIECE_VALS = [0, 1, 3, 4, 5, 6, 8, 10, 0, 7, 7, 7, 7, 11, 13, 0, 0]
_HAND_VALS  = [1, 3, 4, 5, 6, 8, 10]  # HAND_PIECES に対応
_MATERIAL_NORM = 100.0   # 正規化係数（この値でMSEが適切なスケールになる）
_MOBILITY_MAX  = 593.0   # 将棋の合法手数の理論最大値


def _compute_aux_labels(board: Board, turn: int) -> dict:
    """
    1局面分の補助タスクラベルを board から自動計算して返す。

    Parameters
    ----------
    board : Board
        cshogi の Board インスタンス（set_psfen / set_hcp 済み）
    turn : int
        手番 (BLACK=0 / WHITE=1)

    Returns
    -------
    dict with keys:
        king_safety : np.ndarray shape (2,) float32  自玉・敵玉密集度 [0,1]
        material    : np.ndarray shape (1,) float32  駒得スコア（正規化）
        mobility    : np.ndarray shape (1,) float32  合法手数（正規化）
        attack      : np.ndarray shape (9,9) float32 自分の利きバイナリ
        threat      : np.ndarray shape (9,9) float32 相手の利きバイナリ
        damage      : np.ndarray shape (9,9) float32 危険にさらされた自駒
    """
    # ---- 1. Material balance ----
    my_v = opp_v = 0.0
    for sq in range(81):
        p  = board.piece(sq)
        pt = board.piece_type(sq)
        if pt == 0 or pt == 8:      # 空マス or 王
            continue
        v = _PIECE_VALS[pt] if pt < len(_PIECE_VALS) else 0
        if (p >> 4) == turn:
            my_v += v
        else:
            opp_v += v
    hands = board.pieces_in_hand
    for i, v in enumerate(_HAND_VALS):
        my_v  += hands[turn][i]     * v
        opp_v += hands[1 - turn][i] * v
    material = np.array([(my_v - opp_v) / _MATERIAL_NORM], dtype=np.float32)

    # ---- 2. King safety (自玉・敵玉の周囲に自分の駒が何マスあるか) ----
    def _king_safety(color: int) -> float:
        ksq = board.king_square(color)
        kx, ky = ksq // 9, ksq % 9
        count = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = kx + dx, ky + dy
                if 0 <= nx < 9 and 0 <= ny < 9:
                    p2  = board.piece(ny + nx * 9)
                    pt2 = board.piece_type(ny + nx * 9)
                    if pt2 != 0 and (p2 >> 4) == color:
                        count += 1.0
        return count / 8.0   # 最大8マス → [0,1] に正規化

    king_safety = np.array(
        [_king_safety(turn), _king_safety(1 - turn)],
        dtype=np.float32
    )

    # ---- 3. Mobility (合法手数) ----
    mobility = np.array(
        [sum(1 for _ in board.legal_moves) / _MOBILITY_MAX],
        dtype=np.float32
    )

    # ---- 4. Attack / Threat / Damage マップ ----
    # pseudo_legal_moves を列挙して手番側・相手側の利きを記録する
    attack = np.zeros((9, 9), dtype=np.float32)   # 自分の利き
    threat = np.zeros((9, 9), dtype=np.float32)   # 相手の利き

    # ✅ 修正後
    # 自分の利き: 現在の手番で pseudo_legal_moves を列挙
    for mv in board.pseudo_legal_moves:
        to_sq = move_to(mv)
        tx, ty = to_sq // 9, to_sq % 9
        attack[tx][ty] = 1.0

    # 相手の利き: push_pass で手番を切り替えてから列挙し、pop_pass で戻す
    board.push_pass()
    for mv in board.pseudo_legal_moves:
        to_sq = move_to(mv)
        tx, ty = to_sq // 9, to_sq % 9
        threat[tx][ty] = 1.0
    board.pop_pass()

    # damage: 自分の駒が相手の利きにさらされているマス
    damage = np.zeros((9, 9), dtype=np.float32)
    for sq in range(81):
        p  = board.piece(sq)
        pt = board.piece_type(sq)
        if pt != 0 and (p >> 4) == turn:
            x, y = sq // 9, sq % 9
            if threat[x][y] > 0.0:
                damage[x][y] = 1.0

    return {
        "king_safety": king_safety,
        "material":    material,
        "mobility":    mobility,
        "attack":      attack,
        "threat":      threat,
        "damage":      damage,
    }


# ---------------------------------------------------------------------------
# 1. 強化学習用: Experience Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """1局面分の強化学習用データ"""
    state:         np.ndarray
    policy_target: np.ndarray
    value_target:  np.ndarray
    old_log_prob:  Optional[np.ndarray] = None
    king_safety:   Optional[np.ndarray] = None
    material:      Optional[np.ndarray] = None
    mobility:      Optional[np.ndarray] = None
    attack_map:    Optional[np.ndarray] = None
    threat_map:    Optional[np.ndarray] = None
    damage_map:    Optional[np.ndarray] = None
    masked_board:  Optional[np.ndarray] = None
    mask_indices:  Optional[np.ndarray] = None


class SumTree:
    def __init__(self, capacity: int):
        self.capacity  = capacity
        self._tree     = np.zeros(2 * capacity, dtype=np.float64)
        self._write    = 0
        self._n_entries = 0

    def _propagate(self, idx: int, delta: float):
        parent = idx // 2
        while parent >= 1:
            self._tree[parent] += delta
            parent //= 2

    def _retrieve(self, idx: int, s: float) -> int:
        while True:
            left, right = 2 * idx, 2 * idx + 1
            if left >= len(self._tree):
                return idx
            if s <= self._tree[left]:
                idx = left
            else:
                s  -= self._tree[left]
                idx = right

    def add(self, priority: float) -> int:
        leaf_idx = self._write + self.capacity
        self.update(leaf_idx, priority)
        self._write     = (self._write + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)
        return leaf_idx

    def update(self, leaf_idx: int, priority: float):
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, s: float) -> Tuple[int, float]:
        leaf_idx = self._retrieve(1, s)
        return leaf_idx, self._tree[leaf_idx]

    def __len__(self):
        return self._n_entries


class PrioritizedReplayBuffer:
    def __init__(self, capacity=500_000, alpha=0.6, beta_start=0.4, beta_frames=1_000_000):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.epsilon     = 1e-5
        self._tree       = SumTree(capacity)
        self._storage: List[Optional[Experience]] = [None] * capacity
        self._frame      = 0
        self._lock       = threading.Lock()

    def _beta(self) -> float:
        frac = min(1.0, self._frame / self.beta_frames)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def add(self, exp: Experience, td_error: float = 1.0):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        with self._lock:
            leaf_idx    = self._tree.add(priority)
            storage_idx = (self._tree._write - 1) % self.capacity
            self._storage[storage_idx] = exp
            self._frame += 1

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        if len(self._tree) < batch_size:
            return [], np.array([]), np.array([])
        beta        = self._beta()
        experiences = []
        indices     = np.empty(batch_size, dtype=np.int32)
        priorities  = np.empty(batch_size, dtype=np.float64)
        segment     = self._tree._tree[1] / batch_size
        with self._lock:
            for i in range(batch_size):
                s = np.random.uniform(segment * i, segment * (i + 1))
                idx, p = self._tree.sample(s)
                experiences.append(self._storage[idx - self.capacity])
                indices[i]    = idx
                priorities[i] = p
        probs      = priorities / self._tree._tree[1]
        is_weights = (len(self._tree) * probs) ** (-beta)
        is_weights /= is_weights.max()
        return experiences, indices, is_weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        with self._lock:
            for idx, tde in zip(indices, td_errors):
                p = (abs(tde) + self.epsilon) ** self.alpha
                self._tree.update(int(idx), p)

    def __len__(self):
        return len(self._tree)


# ---------------------------------------------------------------------------
# 2. イミテーション学習用: 教師データ読み込み (PSV/HCPE)
# ---------------------------------------------------------------------------

def _score_to_value(score: int, turn: int) -> float:
    """cp 評価値を勝率教師 [0, 1] に変換"""
    v = 1.0 / (1.0 + math.exp(-float(score) / _PSV_SCORE_SCALE))
    return v if turn == BLACK else 1.0 - v


class ShogiDataLoader:
    """
    PSV または HCPE 形式のファイルを読み込む統合データローダー。

    __next__ は同期バッチ生成。aux_labels=True のとき _compute_aux_labels
    を呼び出して補助タスクのラベルも同時に返す。

    Returns (per batch)
    -------------------
    aux_labels=False (デフォルト互換モード):
        (x, move_label, result)
        x          : FloatTensor (B, CH, 9, 9)
        move_label : LongTensor  (B,)
        result     : FloatTensor (B, 1)

    aux_labels=True:
        (x, move_label, result, aux)
        aux は dict[str, Tensor] — 各キーのshapeは上記 _compute_aux_labels 参照
    """

    def __init__(
        self,
        files,
        batch_size,
        device,
        format       = "psv",
        shuffle      = True,
        features_mode = 0,
        aux_labels   = False,   # ← True にすると補助タスクラベルも返す
    ):
        self.batch_size        = batch_size
        self.device            = device
        self.shuffle           = shuffle
        self.format            = format.lower()
        self.aux_labels        = aux_labels
        self.features_settings = FEATURES_SETTINGS[features_mode]

        if isinstance(files, str):
            files = [files]
        self.data = self._load_files(files)

        if self.shuffle:
            np.random.shuffle(self.data)

        self.i     = 0
        self.board = Board()

    def _load_files(self, files):
        data_list = []
        dtype = PackedSfenValue if self.format == "psv" else HuffmanCodedPosAndEval
        for f in files:
            if os.path.exists(f):
                _logger.info(f"Loading {self.format.upper()} data: {f}")
                data_list.append(np.fromfile(f, dtype=dtype))
            else:
                _logger.warning(f"File not found, skipping: {f}")
        if not data_list:
            raise FileNotFoundError(f"No data files found: {files}")
        return np.concatenate(data_list)

    def _make_batch(self, subset):
        """
        subset のレコード群からバッチを生成する。
        aux_labels=True のとき補助ラベルも計算する。
        """
        ch     = self.features_settings.features_num
        b_size = len(subset)

        x = np.zeros((b_size, ch, 9, 9), dtype=np.float32)
        p = np.zeros(b_size,             dtype=np.int64)
        v = np.zeros((b_size, 1),        dtype=np.float32)

        # aux バッファ（aux_labels=True のときのみ確保）
        if self.aux_labels:
            aux_buf = {
                "king_safety": np.zeros((b_size, 2),    dtype=np.float32),
                "material":    np.zeros((b_size, 1),    dtype=np.float32),
                "mobility":    np.zeros((b_size, 1),    dtype=np.float32),
                "attack":      np.zeros((b_size, 1, 9, 9), dtype=np.float32),
                "threat":      np.zeros((b_size, 1, 9, 9), dtype=np.float32),
                "damage":      np.zeros((b_size, 1, 9, 9), dtype=np.float32),
            }

        for idx, entry in enumerate(subset):
            if self.format == "psv":
                self.board.set_psfen(entry["sfen"])
                move16 = move16_from_psv(int(entry["move"]))
                score  = entry["score"]
            else:
                self.board.set_hcp(entry["hcp"])
                move16 = entry["bestMove16"]
                score  = entry["eval"]

            turn = self.board.turn
            self.features_settings.make_features(self.board, x[idx])
            p[idx]    = make_move_label(move16, turn)
            v[idx, 0] = _score_to_value(score, turn)

            if self.aux_labels:
                lbl = _compute_aux_labels(self.board, turn)
                aux_buf["king_safety"][idx]       = lbl["king_safety"]
                aux_buf["material"][idx]          = lbl["material"]
                aux_buf["mobility"][idx]          = lbl["mobility"]
                aux_buf["attack"][idx, 0]         = lbl["attack"]
                aux_buf["threat"][idx, 0]         = lbl["threat"]
                aux_buf["damage"][idx, 0]         = lbl["damage"]

        # numpy → Tensor に変換してデバイスに転送
        x_t = torch.from_numpy(x).to(self.device)
        p_t = torch.from_numpy(p).to(self.device)
        v_t = torch.from_numpy(v).to(self.device)

        if not self.aux_labels:
            return x_t, p_t, v_t

        aux_t = {
            k: torch.from_numpy(arr).to(self.device)
            for k, arr in aux_buf.items()
        }
        return x_t, p_t, v_t, aux_t

    def sample(self):
        """ランダムに1バッチ分サンプリングして返す（バリデーション用）"""
        indices = np.random.randint(0, len(self.data), self.batch_size)
        return self._make_batch(self.data[indices])

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        return self

    def __next__(self):
        if self.i >= len(self.data):
            raise StopIteration
        end        = min(self.i + self.batch_size, len(self.data))
        batch      = self._make_batch(self.data[self.i:end])
        self.i     = end
        return batch

    def __len__(self):
        return len(self.data)
