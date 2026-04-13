"""
buffer.py  —  優先度付き経験再生バッファ (Prioritized Experience Replay)
==========================================================================

設計方針
--------
* SumTree を使って O(log N) でサンプリング・更新を実現する。
  naïve な配列実装は capacity=500k 規模で sample() が極端に遅くなるため
  SumTree は必須。
* 経験は numpy 配列としてバッファに格納し、サンプリング後に
  呼び出し元で Tensor 化する（GPU 転送はトレーニングループ側に任せる）。
* スレッドセーフではない。自己対局スレッドと学習スレッドを
  別々に走らせる場合は外側で Lock を取ること。

格納する経験 (Experience) の構造
----------------------------------
  state       : np.float32  (C, 9, 9)   — 入力盤面特徴
  policy_target: np.float32 (num_actions,) — MCTS 訪問回数を正規化した確率
  value_target : np.float32 (1,)          — 終局報酬 z ∈ {-1, 0, +1}
  old_log_prob : np.float32 (1,)          — 自己対局時の log π(a|s)（PPO 用）
  king_safety  : np.float32 (2,)          — 自玉・敵玉の安全度スカラー
  material     : np.float32 (1,)          — 駒得スカラー
  mobility     : np.float32 (1,)          — 合法手数スカラー
  attack_map   : np.float32 (1, 9, 9)    — 自分の利きマップ
  threat_map   : np.float32 (1, 9, 9)    — 相手の利きマップ
  damage_map   : np.float32 (1, 9, 9)    — 駒が取られるリスクマップ
  masked_board : np.int64   (9, 9)        — マスク再構成ターゲット（駒種クラス）
  mask_indices : np.int64   (K,)          — マスクした升目のインデックス
"""

import numpy as np
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# データクラス: 1 局面分の経験
# ---------------------------------------------------------------------------
@dataclass
class Experience:
    state:         np.ndarray   # float32 (C, 9, 9)
    policy_target: np.ndarray   # float32 (num_actions,)
    value_target:  np.ndarray   # float32 (1,)
    old_log_prob:  np.ndarray   # float32 (1,)  PPO 用
    king_safety:   np.ndarray   # float32 (2,)
    material:      np.ndarray   # float32 (1,)
    mobility:      np.ndarray   # float32 (1,)
    attack_map:    np.ndarray   # float32 (1, 9, 9)
    threat_map:    np.ndarray   # float32 (1, 9, 9)
    damage_map:    np.ndarray   # float32 (1, 9, 9)
    masked_board:  np.ndarray   # int64   (9, 9)
    mask_indices:  np.ndarray   # int64   (K,)


# ---------------------------------------------------------------------------
# SumTree: 優先度の累積和を O(log N) で管理するセグメント木
# ---------------------------------------------------------------------------
class SumTree:
    """
    葉ノードに優先度を格納するセグメント木。
    インデックス規則:
      - 内部ノード: 1 .. capacity-1
      - 葉ノード  : capacity .. 2*capacity-1
    0 番インデックスは使用しない（1-indexed 実装）。
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # ツリー全体（内部ノード + 葉）のサイズ
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._write = 0          # 次に書き込む葉のオフセット (0-indexed)
        self._n_entries = 0      # 実際に格納されている経験数

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    def _propagate(self, idx: int, delta: float):
        """葉 idx の変化量 delta を根まで伝播する。"""
        parent = idx // 2
        while parent >= 1:
            self._tree[parent] += delta
            parent //= 2

    def _retrieve(self, idx: int, s: float) -> int:
        """累積和 s に対応する葉インデックスを返す。"""
        while True:
            left  = 2 * idx
            right = left + 1
            if left >= len(self._tree):   # 葉に到達
                return idx
            if s <= self._tree[left]:
                idx = left
            else:
                s -= self._tree[left]
                idx = right

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------
    @property
    def total_priority(self) -> float:
        return float(self._tree[1])   # 根ノード = 全優先度の総和

    def add(self, priority: float) -> int:
        """優先度を追加し、葉の絶対インデックスを返す。"""
        leaf_idx = self._write + self.capacity
        self.update(leaf_idx, priority)
        self._write = (self._write + 1) % self.capacity
        self._n_entries = min(self._n_entries + 1, self.capacity)
        return leaf_idx

    def update(self, leaf_idx: int, priority: float):
        """葉 leaf_idx の優先度を priority に更新する。"""
        delta = priority - self._tree[leaf_idx]
        self._tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, s: float) -> Tuple[int, float]:
        """
        累積和 s に対応する (葉インデックス, 優先度) を返す。
        s は [0, total_priority) の範囲で渡すこと。
        """
        leaf_idx = self._retrieve(1, s)
        return leaf_idx, self._tree[leaf_idx]

    def __len__(self) -> int:
        return self._n_entries


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    """
    優先度付き経験再生バッファ。

    Parameters
    ----------
    capacity : int
        最大格納件数。古い経験は上書きされる（環状バッファ）。
    alpha : float
        優先度の指数。0 = 均一サンプリング、1 = 優先度に完全比例。
        推奨値 0.6。
    beta_start : float
        IS 重み補正の初期値。学習中に 1.0 へ線形アニールする。
        推奨値 0.4。
    beta_frames : int
        beta が 1.0 に達するまでのフレーム数（更新ステップ数）。
    epsilon : float
        優先度がゼロになるのを防ぐ微小値。
    """

    def __init__(
        self,
        capacity:    int   = 500_000,
        alpha:       float = 0.6,
        beta_start:  float = 0.4,
        beta_frames: int   = 1_000_000,
        epsilon:     float = 1e-5,
    ):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.epsilon     = epsilon

        self._tree      = SumTree(capacity)
        self._storage: List[Optional[Experience]] = [None] * capacity
        # 葉インデックス → ストレージインデックスの逆引き用
        # SumTree の葉インデックス = capacity + write_ptr なので
        # ストレージインデックスは write_ptr と同じ。
        self._frame = 0     # add() の呼ばれた累計回数（beta アニール用）
        self._lock  = threading.Lock()

    # ------------------------------------------------------------------
    # beta のアニール
    # ------------------------------------------------------------------
    def _beta(self) -> float:
        """現在の beta 値（beta_start から 1.0 へ線形増加）。"""
        frac = min(1.0, self._frame / self.beta_frames)
        return self.beta_start + frac * (1.0 - self.beta_start)

    # ------------------------------------------------------------------
    # 優先度の計算
    # ------------------------------------------------------------------
    def _priority(self, td_error: float) -> float:
        return (abs(td_error) + self.epsilon) ** self.alpha

    # ------------------------------------------------------------------
    # 経験の追加
    # ------------------------------------------------------------------
    def add(self, exp: Experience, td_error: float = 1.0):
        """
        経験を追加する。

        Parameters
        ----------
        exp : Experience
            格納する経験。
        td_error : float
            この経験の初期 TD 誤差。新しい経験にはバッファ内最大優先度
            を与えるのが一般的だが、ここでは td_error=1.0 をデフォルトと
            して学習開始時に均一に近い優先度にする。
        """
        priority = self._priority(td_error)
        with self._lock:
            leaf_idx   = self._tree.add(priority)
            storage_idx = (self._tree._write - 1) % self.capacity
            # _write はすでに +1 されているため -1 で現在の書き込み位置を取得
            self._storage[storage_idx] = exp
            self._frame += 1

    def add_batch(self, experiences: List[Experience], td_errors: Optional[np.ndarray] = None):
        """
        複数の経験をまとめて追加する（自己対局1ゲーム分を一括投入する際に使う）。
        """
        if td_errors is None:
            td_errors = np.ones(len(experiences), dtype=np.float32)
        for exp, tde in zip(experiences, td_errors):
            self.add(exp, float(tde))

    # ------------------------------------------------------------------
    # サンプリング
    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        優先度比例サンプリングを行い、IS 重みと共にバッチを返す。

        Returns
        -------
        experiences : List[Experience]  長さ batch_size
        indices     : np.ndarray int32  (batch_size,)  SumTree 葉インデックス
        is_weights  : np.ndarray float32 (batch_size,) IS 補正重み (正規化済み)
        """
        assert len(self) >= batch_size, (
            f"バッファに十分なデータがありません: {len(self)} < {batch_size}"
        )

        beta = self._beta()
        experiences: List[Experience] = []
        indices     = np.empty(batch_size, dtype=np.int32)
        priorities  = np.empty(batch_size, dtype=np.float64)

        # 優先度区間を batch_size 等分してサンプリング（Stratified sampling）
        segment = self._tree.total_priority / batch_size

        with self._lock:
            for i in range(batch_size):
                lo = segment * i
                hi = segment * (i + 1)
                s  = np.random.uniform(lo, hi)
                leaf_idx, priority = self._tree.sample(s)

                # 葉インデックス → ストレージインデックス
                storage_idx = leaf_idx - self.capacity
                exp = self._storage[storage_idx]

                # None の場合（未書き込み領域）は再サンプリング
                retry = 0
                while exp is None and retry < 10:
                    s = np.random.uniform(0, self._tree.total_priority)
                    leaf_idx, priority = self._tree.sample(s)
                    storage_idx = leaf_idx - self.capacity
                    exp = self._storage[storage_idx]
                    retry += 1

                experiences.append(exp)
                indices[i]    = leaf_idx
                priorities[i] = priority

        # IS 重みの計算: w_i = (N * P(i))^{-beta} / max_w
        n = len(self)
        probs      = priorities / self._tree.total_priority
        is_weights = (n * probs) ** (-beta)
        is_weights /= is_weights.max()   # 最大値で正規化（最大 = 1.0）

        return experiences, indices, is_weights.astype(np.float32)

    # ------------------------------------------------------------------
    # 優先度の更新（学習後に TD 誤差で更新）
    # ------------------------------------------------------------------
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        学習後に各サンプルの TD 誤差で優先度を更新する。

        Parameters
        ----------
        indices   : SumTree 葉インデックス（sample() の戻り値そのまま）
        td_errors : 絶対 TD 誤差  shape (batch_size,)
        """
        with self._lock:
            for idx, tde in zip(indices, td_errors):
                priority = self._priority(float(tde))
                self._tree.update(int(idx), priority)

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._tree)

    def is_ready(self, min_size: int) -> bool:
        """min_size 以上のデータが蓄積されていれば True。"""
        return len(self) >= min_size

    # ------------------------------------------------------------------
    # チェックポイント保存・ロード
    # ------------------------------------------------------------------
    def save(self, path: str):
        """
        バッファの状態を .npz 形式で保存する。
        大きなバッファは数 GB になるので、保存先のディスク容量に注意。
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'storage':     self._storage,
                'tree':        self._tree._tree,
                'write':       self._tree._write,
                'n_entries':   self._tree._n_entries,
                'frame':       self._frame,
                'alpha':       self.alpha,
                'beta_start':  self.beta_start,
                'beta_frames': self.beta_frames,
                'capacity':    self.capacity,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Buffer] Saved {len(self)} experiences → {path}")

    def load(self, path: str):
        """保存されたバッファ状態をロードする。"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        assert data['capacity'] == self.capacity, \
            f"capacity mismatch: {data['capacity']} vs {self.capacity}"
        self._storage            = data['storage']
        self._tree._tree         = data['tree']
        self._tree._write        = data['write']
        self._tree._n_entries    = data['n_entries']
        self._frame              = data['frame']
        print(f"[Buffer] Loaded {len(self)} experiences ← {path}")


# ---------------------------------------------------------------------------
# ユーティリティ: Experience バッチ → テンソル辞書に変換
# ---------------------------------------------------------------------------
def collate_experiences(
    experiences: List[Experience],
    device: "torch.device",  # noqa: F821
) -> dict:
    """
    List[Experience] を PyTorch Tensor の辞書に変換する。
    train.py の学習ループ内でバッチ化に使う。

    Returns
    -------
    dict of str → torch.Tensor  (全て device 上)
    """
    import torch

    def stack(attr: str, dtype=torch.float32) -> "torch.Tensor":
        arr = np.stack([getattr(e, attr) for e in experiences], axis=0)
        return torch.tensor(arr, dtype=dtype, device=device)

    return {
        # モデル入力
        'state':          stack('state'),               # (B, C, 9, 9)
        # RL ターゲット
        'policy_target':  stack('policy_target'),       # (B, num_actions)
        'value_target':   stack('value_target'),        # (B, 1)
        'old_log_prob':   stack('old_log_prob'),        # (B, 1)
        # 補助タスク教師
        'king_safety':    stack('king_safety'),         # (B, 2)
        'material':       stack('material'),            # (B, 1)
        'mobility':       stack('mobility'),            # (B, 1)
        'attack_map':     stack('attack_map'),          # (B, 1, 9, 9)
        'threat_map':     stack('threat_map'),          # (B, 1, 9, 9)
        'damage_map':     stack('damage_map'),          # (B, 1, 9, 9)
        # 自己教師あり
        'masked_board':   stack('masked_board', dtype=torch.long),  # (B, 9, 9)
        'mask_indices':   stack('mask_indices', dtype=torch.long),  # (B, K)
    }
