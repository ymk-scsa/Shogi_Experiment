"""
buffer.py — データ読み込みと経験再生バッファの統合
=================================================

このモジュールは以下の2つの機能を提供します：
1. 教師データ（PSV/HCPE）の読み込み（イミテーション学習用）
2. 自己対局データの管理（強化学習用: Prioritized Experience Replay）

教師データとして Suisho10Mn_psv.bin などの PackedSfenValue 形式をサポートします。
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
from cshogi import Board, HuffmanCodedPosAndEval, PackedSfenValue, move16_from_psv, BLACK

# プロジェクトルートをパスに追加（features_settings 等のインポート用）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from game.board import FEATURES_SETTINGS, make_move_label, make_result

_logger = logging.getLogger(__name__)
_PSV_SCORE_SCALE = 600.0

# ---------------------------------------------------------------------------
# 1. 強化学習用: Experience Replay Buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """1局面分の強化学習用データ"""
    state:         np.ndarray   # float32 (C, 9, 9)
    policy_target: np.ndarray   # float32 (num_actions,) - MCTS訪問確率
    value_target:  np.ndarray   # float32 (1,) - 終局報酬 [-1, 1]
    # PPO用
    old_log_prob:  Optional[np.ndarray] = None # (1,)
    # 補助タスク用
    king_safety:   Optional[np.ndarray] = None # (2,)
    material:      Optional[np.ndarray] = None # (1,)
    mobility:      Optional[np.ndarray] = None # (1,)
    attack_map:    Optional[np.ndarray] = None # (1, 9, 9)
    threat_map:    Optional[np.ndarray] = None # (1, 9, 9)
    damage_map:    Optional[np.ndarray] = None # (1, 9, 9)
    # SSL用
    masked_board:  Optional[np.ndarray] = None # (9, 9)
    mask_indices:  Optional[np.ndarray] = None # (K,)

class SumTree:
    """優先度付きサンプリングのためのセグメント木"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._write = 0
        self._n_entries = 0

    def _propagate(self, idx: int, delta: float):
        parent = idx // 2
        while parent >= 1:
            self._tree[parent] += delta
            parent //= 2

    def _retrieve(self, idx: int, s: float) -> int:
        while True:
            left = 2 * idx
            right = left + 1
            if left >= len(self._tree): return idx
            if s <= self._tree[left]: idx = left
            else:
                s -= self._tree[left]
                idx = right

    def add(self, priority: float) -> int:
        leaf_idx = self._write + self.capacity
        self.update(leaf_idx, priority)
        self._write = (self._write + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)
        return leaf_idx

    def update(self, leaf_idx: int, priority: float):
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, s: float) -> Tuple[int, float]:
        leaf_idx = self._retrieve(1, s)
        return leaf_idx, self._tree[leaf_idx]

    def __len__(self): return self._n_entries

class PrioritizedReplayBuffer:
    """優先度付き経験再生バッファ"""
    def __init__(self, capacity=500_000, alpha=0.6, beta_start=0.4, beta_frames=1_000_000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = 1e-5
        self._tree = SumTree(capacity)
        self._storage: List[Optional[Experience]] = [None] * capacity
        self._frame = 0
        self._lock = threading.Lock()

    def _beta(self) -> float:
        frac = min(1.0, self._frame / self.beta_frames)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def add(self, exp: Experience, td_error: float = 1.0):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        with self._lock:
            leaf_idx = self._tree.add(priority)
            storage_idx = (self._tree._write - 1) % self.capacity
            self._storage[storage_idx] = exp
            self._frame += 1

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        if len(self._tree) < batch_size: return [], np.array([]), np.array([])
        
        beta = self._beta()
        experiences = []
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)
        segment = self._tree._tree[1] / batch_size

        with self._lock:
            for i in range(batch_size):
                s = np.random.uniform(segment * i, segment * (i + 1))
                idx, p = self._tree.sample(s)
                experiences.append(self._storage[idx - self.capacity])
                indices[i] = idx
                priorities[i] = p

        probs = priorities / self._tree._tree[1]
        is_weights = (len(self._tree) * probs) ** (-beta)
        is_weights /= is_weights.max()
        return experiences, indices, is_weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        with self._lock:
            for idx, tde in zip(indices, td_errors):
                p = (abs(tde) + self.epsilon) ** self.alpha
                self._tree.update(int(idx), p)

    def __len__(self): return len(self._tree)

# ---------------------------------------------------------------------------
# 2. イミテーション学習用: 教師データ読み込み (PSV/HCPE)
# ---------------------------------------------------------------------------

def _score_to_value(score: int, turn: int) -> float:
    """cp評価値を勝率教師 [0, 1] に変換"""
    v = 1.0 / (1.0 + math.exp(-float(score) / _PSV_SCORE_SCALE))
    return v if turn == BLACK else 1.0 - v

class ShogiDataLoader:
    """PSV または HCPE 形式のファイルを読み込む統合データローダー"""
    def __init__(self, files, batch_size, device, format="psv", shuffle=True, features_mode=0):
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.format = format.lower()
        self.features_settings = FEATURES_SETTINGS[features_mode]
        
        if isinstance(files, str): files = [files]
        self.data = self._load_files(files)
        
        if self.shuffle:
            np.random.shuffle(self.data)
            
        self.i = 0
        self.board = Board()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_future = None

    def _load_files(self, files):
        data_list = []
        dtype = PackedSfenValue if self.format == "psv" else HuffmanCodedPosAndEval
        for f in files:
            if os.path.exists(f):
                _logger.info(f"Loading {self.format.upper()} data: {f}")
                data_list.append(np.fromfile(f, dtype=dtype))
        if not data_list:
            raise FileNotFoundError(f"No data files found for {files}")
        return np.concatenate(data_list)

    def _make_batch(self, subset):
        ch = self.features_settings.features_num
        b_size = len(subset)
        x = np.zeros((b_size, ch, 9, 9), dtype=np.float32)
        p = np.zeros(b_size, dtype=np.int64)
        v = np.zeros((b_size, 1), dtype=np.float32)
        
        for idx, entry in enumerate(subset):
            if self.format == "psv":
                self.board.set_psfen(entry["sfen"])
                move16 = move16_from_psv(int(entry["move"]))
                score = entry["score"]
            else:
                self.board.set_hcp(entry["hcp"])
                move16 = entry["bestMove16"]
                score = entry["eval"] # HCPEのevalをscoreとして扱う
            
            self.features_settings.make_features(self.board, x[idx])
            p[idx] = make_move_label(move16, self.board.turn)
            v[idx, 0] = _score_to_value(score, self.board.turn)
            
        return (torch.from_numpy(x).to(self.device), 
                torch.from_numpy(p).to(self.device), 
                torch.from_numpy(v).to(self.device))

    def sample(self):
        """ランダムに1バッチ分サンプリングして返す"""
        indices = np.random.randint(0, len(self.data), self.batch_size)
        return self._make_batch(self.data[indices])

    def __iter__(self):
        self.i = 0
        if self.shuffle: np.random.shuffle(self.data)
        return self

    def __next__(self):
        if self.i >= len(self.data): raise StopIteration
        end = min(self.i + self.batch_size, len(self.data))
        batch = self._make_batch(self.data[self.i:end])
        self.i = end
        return batch

    def __len__(self): return len(self.data)
