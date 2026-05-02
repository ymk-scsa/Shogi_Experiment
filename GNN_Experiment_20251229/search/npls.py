"""
search/npls.py
UW-NPLS (Uncertainty-Weighted Neural Priority Leaf Search) の実装。

主な改善点：
  - 全バグ修正（first_move条件逆・Transposition Table・heapq比較・再展開防止）
  - バッチ推論対応（GPUを効率的に使用）
  - Uncertainty を探索部側で計算（評価関数依存から脱却）
      Strategy A: 子ノードvalue分散
      Strategy B: 訪問回数逆数
      Strategy C: 子ノード評価の不一致度（max - min）
  - SPSAパラメータ調整インターフェース対応
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
import math
import sys
import os
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict

from cshogi import (
    Board,
    BLACK,
    move_to_usi,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from game.board import (
    FEATURES_SETTINGS,
    MOVE_LABELS_NUM,
    make_move_label,
)
from model.model import create_model


# ===========================================================
# 定数
# ===========================================================

DEFAULT_GPU_ID            = 0
# バッチサイズ：GPUに一度に投げる局面数。
# 大きいほどGPUスループットが上がるが、メモリ使用量も増える。
# NPLSでは優先度キューから複数ノードをまとめてpopしてバッチ推論する。
DEFAULT_BATCH_SIZE        = 64
DEFAULT_RESIGN_THRESHOLD  = 0.01   # 勝率1%以下で投了
DEFAULT_BYOYOMI_MARGIN    = 100    # 秒読みマージン(ms)：通信遅延の余裕
DEFAULT_CONST_NODES       = 1000   # 時間指定なし時の探索ノード数上限
DEFAULT_BLOCKS_CONFIG     = "30blocks"

# NPLS 優先度重みパラメータ
DEFAULT_ALPHA       = 1.0   # Q(value)の重み
DEFAULT_BETA        = 0.5   # policy の重み
DEFAULT_GAMMA       = 0.2   # depth補正の重み
DEFAULT_DELTA       = 0.3   # uncertainty の重み（探索部計算）
DEFAULT_EPSILON     = 0.5   # 探索ボーナス 1/√visit の重み
DEFAULT_ZETA        = 0.7   # UW bonus（uncertainty×visit減衰）の重み

# RootMoveStats スコアリングパラメータ
DEFAULT_W_MAX       = 0.40  # 最大値の重み
DEFAULT_W_MEAN      = 0.40  # uncertainty-weighted平均の重み
DEFAULT_W_COUNT     = 0.10  # 探索回数の重み
DEFAULT_VAR_PENALTY = 0.10  # 評価分散のペナルティ重み

# 探索部uncertainty ミキシング重み（合計1.0推奨）
DEFAULT_UNC_W_VAR   = 0.5   # Strategy A: 子ノードvalue分散
DEFAULT_UNC_W_VISIT = 0.3   # Strategy B: 訪問回数逆数
DEFAULT_UNC_W_RANGE = 0.2   # Strategy C: 子ノードvalue範囲(max-min)


# ===========================================================
# NPLSNode
# ===========================================================

@dataclass
class NPLSNode:
    """探索ノード。heapq用のpriorityはカウンタで比較するため__lt__不要。"""
    state_sfen:  str
    first_move:  Optional[int]
    value:       float
    policy:      float
    depth:       int
    visit_count: int   = 1
    total_value: float = 0.0
    # 探索部で計算するuncertainty（NNからは受け取らない）
    uncertainty: float = 0.0
    priority:    float = 0.0

    def update_value(self, new_value: float) -> None:
        self.visit_count += 1
        self.total_value += new_value
        self.value = self.total_value / self.visit_count


# ===========================================================
# SearchUncertaintyTracker
# ===========================================================

class SearchUncertaintyTracker:
    """
    探索部側でuncertaintyを計算するクラス。
    評価関数（NN）に依存しない。

    3戦略を混合する：
      A: 子ノードvalueの分散（探索が深まるほど正確）
      B: 訪問回数逆数（未探索ほど不確実）
      C: 子ノードvalueの範囲（tactical swingの大きい局面を検出）
    """

    def __init__(
        self,
        w_var:   float = DEFAULT_UNC_W_VAR,
        w_visit: float = DEFAULT_UNC_W_VISIT,
        w_range: float = DEFAULT_UNC_W_RANGE,
    ):
        self.w_var   = w_var
        self.w_visit = w_visit
        self.w_range = w_range

        # sfen → 子ノードvalue リスト
        self._child_values: Dict[str, List[float]] = defaultdict(list)

    def register_child(self, parent_sfen: str, child_value: float) -> None:
        """子ノードを登録する（展開時に呼ぶ）"""
        self._child_values[parent_sfen].append(child_value)

    def compute(self, node: NPLSNode) -> float:
        """
        ノードのuncertaintyを3戦略で計算して返す。

        Parameters
        ----------
        node : NPLSNode

        Returns
        -------
        float : uncertainty（0.0〜1.0程度）
        """
        # Strategy A: 子ノードvalue分散
        children = self._child_values.get(node.state_sfen, [])
        if len(children) >= 2:
            arr = np.array(children, dtype=np.float32)
            unc_var = float(np.var(arr))
        else:
            unc_var = 1.0  # 子ノードが少ない = 不確実

        # Strategy B: 訪問回数逆数（未探索ほど高い）
        unc_visit = 1.0 / math.sqrt(node.visit_count + 1)

        # Strategy C: 子ノードvalue範囲（max - min）
        if len(children) >= 2:
            unc_range = float(max(children) - min(children))
        else:
            unc_range = 1.0

        # 合計で割って正規化：重みの比率だけSPSAが調整する
        # → deltaとのスケール関係が安定する
        total_w = self.w_var + self.w_visit + self.w_range + 1e-8
        uncertainty = (
            self.w_var   * unc_var
            + self.w_visit * unc_visit
            + self.w_range * unc_range
        ) / total_w

        return float(np.clip(uncertainty, 0.0, 10.0))

    def reset(self) -> None:
        self._child_values.clear()


# ===========================================================
# RootMoveStats（UW版）
# ===========================================================

class RootMoveStats:
    """
    初手ごとの探索統計。
    KataGo的 Uncertainty-Weighted 平均と分散ペナルティを導入。
    """

    def __init__(
        self,
        w_max:       float = DEFAULT_W_MAX,
        w_mean:      float = DEFAULT_W_MEAN,
        w_count:     float = DEFAULT_W_COUNT,
        var_penalty: float = DEFAULT_VAR_PENALTY,
    ):
        # SPSAが更新した重みをインスタンス変数に保存
        self._w_max       = w_max
        self._w_mean      = w_mean
        self._w_count     = w_count
        self._var_penalty = var_penalty
        self.stats: Dict[int, Dict[str, float]] = defaultdict(lambda: {
            "max_value":    -float("inf"),
            "sum_value":    0.0,
            "sum_sq":       0.0,
            "count":        0,
            "weighted_sum": 0.0,
            "weight_total": 0.0,
        })

    def update(self, move: int, value: float, uncertainty: float) -> None:
        if move is None:
            return
        s = self.stats[move]
        s["max_value"] = max(s["max_value"], value)
        s["sum_value"] += value
        s["sum_sq"]    += value * value
        s["count"]     += 1
        # uncertainty が小さいほど（安定しているほど）重みを大きく
        weight = 1.0 / (uncertainty ** 2 + 1e-8)
        s["weighted_sum"]  += weight * value
        s["weight_total"]  += weight

    def variance(self, move: int) -> float:
        s = self.stats[move]
        n = s["count"]
        if n < 2:
            return 0.0
        mean = s["sum_value"] / n
        return max(s["sum_sq"] / n - mean ** 2, 0.0)

    def score(self, move: int) -> float:
        s = self.stats[move]
        if s["count"] == 0:
            return -float("inf")
        wt = s["weight_total"]
        weighted_mean = s["weighted_sum"] / (wt + 1e-8) if wt > 0 else 0.0
        # w_max + w_mean + w_count を正規化してスケール安定化
        total_w = self._w_max + self._w_mean + self._w_count + 1e-8
        return (
            (self._w_max   * s["max_value"]
             + self._w_mean  * weighted_mean
             + self._w_count * math.log(s["count"] + 1)) / total_w
            - self._var_penalty * self.variance(move)
        )

    def best_move(self) -> Optional[int]:
        if not self.stats:
            return None
        return max(self.stats.keys(), key=self.score)


# ===========================================================
# NPLSPlayer
# ===========================================================

class NPLSPlayer:
    """
    USI将棋エンジン。
    UW-NPLS（Uncertainty-Weighted Neural Priority Leaf Search）を使用。
    """

    name = "GNN-Shogi-NPLS"
    DEFAULT_MODELFILE = "weights/checkpoint.pth"

    def __init__(
        self,
        features_mode:      int = 0,
        blocks_config_mode: str = DEFAULT_BLOCKS_CONFIG,
        name:               str = "GNN-Shogi-NPLS",
        modelfile:          str = "weights/checkpoint.pth",
    ) -> None:
        self.name               = name
        self.modelfile          = modelfile
        self.model              = None
        self.device             = None
        self.gpu_id             = DEFAULT_GPU_ID
        self.batch_size         = DEFAULT_BATCH_SIZE

        self.features_setting   = FEATURES_SETTINGS[features_mode]
        self.blocks_config_mode = blocks_config_mode
        self.debug              = False

        self.root_board         = Board()
        self.halt               = None
        self.time_limit_ms      = 0.0
        self.minimum_time_ms    = 0.0

        # NPLS優先度パラメータ
        self.alpha       = DEFAULT_ALPHA
        self.beta        = DEFAULT_BETA
        self.gamma       = DEFAULT_GAMMA
        self.delta       = DEFAULT_DELTA
        self.epsilon     = DEFAULT_EPSILON
        self.zeta        = DEFAULT_ZETA

        # RootMoveStatsスコアリングパラメータ（全てSPSA対象）
        self.w_max       = DEFAULT_W_MAX
        self.w_mean      = DEFAULT_W_MEAN
        self.w_count     = DEFAULT_W_COUNT
        self.var_penalty = DEFAULT_VAR_PENALTY
        # w_max + w_mean + w_count は正規化するので合計!=1でもOK

        # 探索部uncertainty ミキシング重み
        self.unc_w_var   = DEFAULT_UNC_W_VAR
        self.unc_w_visit = DEFAULT_UNC_W_VISIT
        self.unc_w_range = DEFAULT_UNC_W_RANGE

    # ----------------------------------------------------------
    # USI プロトコル
    # ----------------------------------------------------------

    def usi(self) -> None:
        print(f"id name {self.name}")
        print(f"option name modelfile type string default {self.DEFAULT_MODELFILE}")
        print(f"option name gpu_id type spin default {DEFAULT_GPU_ID} min -1 max 7")
        print(f"option name batch_size type spin default {DEFAULT_BATCH_SIZE} min 1 max 512")
        # 優先度重みパラメータ（×100で整数化）
        for pname, default in [
            ("alpha",       int(DEFAULT_ALPHA       * 100)),
            ("beta",        int(DEFAULT_BETA        * 100)),
            ("gamma",       int(DEFAULT_GAMMA       * 100)),
            ("delta",       int(DEFAULT_DELTA       * 100)),
            ("epsilon",     int(DEFAULT_EPSILON     * 100)),
            ("zeta",        int(DEFAULT_ZETA        * 100)),
            ("w_max",       int(DEFAULT_W_MAX       * 100)),  # 追加
            ("w_mean",      int(DEFAULT_W_MEAN      * 100)),  # 追加
            ("w_count",     int(DEFAULT_W_COUNT     * 100)),  # 追加
            ("var_penalty", int(DEFAULT_VAR_PENALTY * 100)),
            ("unc_w_var",   int(DEFAULT_UNC_W_VAR   * 100)),
            ("unc_w_visit", int(DEFAULT_UNC_W_VISIT * 100)),
            ("unc_w_range", int(DEFAULT_UNC_W_RANGE * 100)),
        ]:
            print(f"option name {pname} type spin default {default} min 0 max 1000")
        print("option name debug type check default false")
        print("usiok")

    def setoption(self, args: list) -> None:
        name = args[1]
        val  = args[3]
        if name == "modelfile":
            self.modelfile = val
        elif name == "gpu_id":
            self.gpu_id = int(val)
        elif name == "batch_size":
            self.batch_size = int(val)
        elif name == "debug":
            self.debug = val == "true"
        elif name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                      "w_max", "w_mean", "w_count", "var_penalty",   # w_max/w_mean/w_count追加
                      "unc_w_var", "unc_w_visit", "unc_w_range"):
            setattr(self, name, int(val) / 100.0)

    def isready(self) -> None:
        self.device = (
            torch.device(f"cuda:{self.gpu_id}")
            if self.gpu_id >= 0 else torch.device("cpu")
        )
        self.model = create_model(
            input_channels=self.features_setting.features_num,
            num_actions=MOVE_LABELS_NUM,
            mode=self.blocks_config_mode,
        )
        self.model.to(self.device)
        if os.path.exists(self.modelfile):
            checkpoint = torch.load(self.modelfile, map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.root_board.reset()
        print("readyok")

    def position(self, sfen: str, usi_moves: list) -> None:
        if sfen == "startpos":
            self.root_board.reset()
        elif sfen.startswith("sfen "):
            self.root_board.set_sfen(sfen[5:])
        for usi_move in usi_moves:
            self.root_board.push_usi(usi_move)

    def set_limits(self, **kwargs) -> None:
        if kwargs.get("infinite") or kwargs.get("ponder"):
            self.halt = 10 ** 9
            return

        if kwargs.get("nodes"):
            self.halt = int(kwargs["nodes"])
            return

        self.halt = None
        btime    = kwargs.get("btime")
        wtime    = kwargs.get("wtime")
        byoyomi  = kwargs.get("byoyomi")
        binc     = kwargs.get("binc", 0)
        winc     = kwargs.get("winc", 0)
        inc      = int(binc) if self.root_board.turn == BLACK else int(winc)
        remaining = int(btime) if self.root_board.turn == BLACK else int(wtime)

        if remaining is None and byoyomi is None:
            self.halt = DEFAULT_CONST_NODES
            return

        remaining = remaining or 0
        self.time_limit_ms  = remaining / 20 + inc
        self.minimum_time_ms = (
            max(int(byoyomi) - DEFAULT_BYOYOMI_MARGIN, 0) if byoyomi else 0
        )
        if self.time_limit_ms < self.minimum_time_ms:
            self.time_limit_ms = self.minimum_time_ms

    # ----------------------------------------------------------
    # 内部ユーティリティ
    # ----------------------------------------------------------

    def _init_search(self) -> None:
        """探索状態を初期化する"""
        self.open_list:           List                  = []
        self.transposition_table: Dict[str, NPLSNode]  = {}
        self.expanded:            set                   = set()
        # RootMoveStatsにw_max/w_mean/w_countを渡す（SPSA反映）
        self.root_stats          = RootMoveStats(
            w_max       = self.w_max,
            w_mean      = self.w_mean,
            w_count     = self.w_count,
            var_penalty = self.var_penalty,
        )
        # SPSAで更新されたunc重みをtracker生成時に反映（問題2の修正）
        self.unc_tracker         = SearchUncertaintyTracker(
            w_var   = self.unc_w_var,
            w_visit = self.unc_w_visit,
            w_range = self.unc_w_range,
        )
        self._push_counter       = itertools.count()
        self.nodes_searched      = 0

    def _push_node(self, node: NPLSNode) -> None:
        """優先度を計算してheapqに追加する"""
        node.priority = self._compute_priority(node)
        heapq.heappush(
            self.open_list,
            (-node.priority, next(self._push_counter), node),
        )

    def _pop_best_node(self) -> NPLSNode:
        return heapq.heappop(self.open_list)[2]

    def _compute_priority(self, node: NPLSNode) -> float:
        Q = node.value
        P = node.policy
        D = math.log(node.depth + 1)
        U = node.uncertainty
        E = 1.0 / math.sqrt(node.visit_count + 1)
        # UW bonus：uncertainty大かつvisit少のノードを積極的に探索
        B = U / (1.0 + math.sqrt(node.visit_count))
        return (
            self.alpha   * Q
            + self.beta  * P
            + self.gamma * D
            + self.delta * U
            + self.epsilon * E
            + self.zeta  * B
        )

    # ----------------------------------------------------------
    # バッチ推論
    # ----------------------------------------------------------

    def _batch_evaluate(
        self,
        boards: List[Board],
    ) -> List[Tuple[float, np.ndarray]]:
        """
        複数局面をまとめてGPUで推論する。
        uncertaintyはNNから受け取らず、探索部で後から計算する。

        Returns
        -------
        List of (value, policy_vector)
        """
        n = len(boards)
        features_batch = torch.zeros(
            (n, self.features_setting.features_num, 9, 9),
            dtype=torch.float32,
        )
        for i, board in enumerate(boards):
            self.features_setting.make_features(
                board, features_batch[i].numpy()
            )

        features_batch = features_batch.to(self.device)

        with torch.no_grad():
            policy_logits, value_tanh = self.model(features_batch)[:2]
            policies = torch.exp(
                F.log_softmax(policy_logits, dim=1)
            ).cpu().numpy()
            values = ((value_tanh.squeeze(1) + 1.0) / 2.0).cpu().numpy()

        return [(float(values[i]), policies[i]) for i in range(n)]

    # ----------------------------------------------------------
    # 展開（バッチ対応）
    # ----------------------------------------------------------

    def _expand_batch(
        self,
        nodes: List[NPLSNode],
    ) -> List[NPLSNode]:
        """
        複数ノードをまとめて展開し、子ノードリストを返す。
        Transposition Table の参照・更新もここで行う。

        探索部uncertaintyの計算：
          - 展開時に子ノードのvalueを unc_tracker に登録
          - 親ノードのuncertaintyを展開後に更新
        """
        # (parent_node, move, next_board) のリストを作成
        # Transposition Tableに既存のものはキューに戻して skip
        to_evaluate: List[Tuple[NPLSNode, int, Board, int]] = []
        # (parent, move, board, current_turn)

        for node in nodes:
            if node.state_sfen in self.expanded:
                continue
            self.expanded.add(node.state_sfen)

            board = Board(node.state_sfen)
            if board.is_game_over():
                continue

            for move in board.legal_moves:
                current_turn = board.turn   # push前に保存（バグ3修正）
                board.push(move)
                child_sfen = board.sfen()

                if child_sfen in self.transposition_table:
                    # バグ2修正：visit_countだけ増やして再キューイング
                    existing = self.transposition_table[child_sfen]
                    existing.visit_count += 1
                    # uncertainty を再計算（動的更新）
                    existing.uncertainty = self.unc_tracker.compute(existing)
                    self._push_node(existing)
                    board.pop()
                    continue

                to_evaluate.append((node, move, Board(child_sfen), current_turn))
                board.pop()

        if not to_evaluate:
            return []

        # バッチ推論（uncertaintyはNNから受け取らない）
        eval_boards = [t[2] for t in to_evaluate]
        results     = self._batch_evaluate(eval_boards)

        children: List[NPLSNode] = []

        for (parent, move, child_board, current_turn), (val, pol_vec) in zip(
            to_evaluate, results
        ):
            move_label   = make_move_label(move, current_turn)  # バグ3修正
            move_policy  = float(pol_vec[move_label])
            child_sfen   = child_board.sfen()

            # バグ1修正：first_move の条件が逆だったのを修正
            first_move = move if parent.depth == 0 else parent.first_move

            child = NPLSNode(
                state_sfen  = child_sfen,
                first_move  = first_move,
                value       = val,
                policy      = move_policy,
                depth       = parent.depth + 1,
                total_value = val,
                visit_count = 1,
                uncertainty = 0.0,  # 初期値。探索が進むと更新される
            )

            self.transposition_table[child_sfen] = child

            # 探索部uncertainty Strategy A/C のためにvalueを登録
            self.unc_tracker.register_child(parent.state_sfen, val)

            children.append((parent, child))

        # 親ノードのuncertaintyを更新（子ノードのvalue分布から計算）
        parent_updated: set = set()
        for parent, child in children:
            if parent.state_sfen not in parent_updated:
                parent.uncertainty = self.unc_tracker.compute(parent)
                parent_updated.add(parent.state_sfen)
            # Strategy B（visit逆数）は child 自身のuncertaintyとして使う
            child.uncertainty = self.unc_tracker.compute(child)

        # RootMoveStats 更新
        for parent, child in children:
            if child.first_move is not None:
                self.root_stats.update(
                    child.first_move,
                    child.value,
                    child.uncertainty,
                )

        return [child for _, child in children]

    # ----------------------------------------------------------
    # 探索メインループ
    # ----------------------------------------------------------

    def go(self) -> Tuple[str, Optional[str]]:
        self._init_search()
        begin_time = time.time()

        if self.root_board.is_game_over():
            return "resign", None

        # ルート局面を評価してキューに投入
        root_results = self._batch_evaluate([self.root_board])
        root_value, root_policy_vec = root_results[0]

        root_node = NPLSNode(
            state_sfen  = self.root_board.sfen(),
            first_move  = None,
            value       = root_value,
            policy      = 1.0,
            depth       = 0,
            total_value = root_value,
            visit_count = 1,
            uncertainty = 1.0,  # 初期は高uncertainty（まだ何も探索していない）
        )
        self._push_node(root_node)

        while True:
            if not self.open_list:
                break

            elapsed_ms = (time.time() - begin_time) * 1000.0

            # 終了判定
            if self.halt is not None:
                if self.nodes_searched >= self.halt:
                    break
            else:
                if elapsed_ms >= self.time_limit_ms:
                    break

            # バッチサイズ分だけノードをpopする
            batch_nodes: List[NPLSNode] = []
            while self.open_list and len(batch_nodes) < self.batch_size:
                node = self._pop_best_node()
                # 再展開防止（バグ5修正）
                if node.state_sfen in self.expanded:
                    continue
                batch_nodes.append(node)

            if not batch_nodes:
                continue

            # バッチ展開
            new_children = self._expand_batch(batch_nodes)
            self.nodes_searched += len(batch_nodes)

            for child in new_children:
                self._push_node(child)

            # デバッグ出力
            if self.debug and self.nodes_searched % 200 == 0:
                elapsed_ms = (time.time() - begin_time) * 1000.0
                best = self.root_stats.best_move()
                best_score = self.root_stats.score(best) if best else 0.0
                print(
                    f"info nodes {self.nodes_searched} "
                    f"time {int(elapsed_ms)} "
                    f"score cp {int(best_score * 1000)}"
                )

        # 最善手の決定
        best_move = self.root_stats.best_move()   # 引数不要になった
        if best_move is None:
            legal = list(self.root_board.legal_moves)
            if not legal:
                return "resign", None
            best_move = legal[0]

        # 投了判定
        best_score = self.root_stats.score(best_move)  # 引数不要になった
        if best_score < DEFAULT_RESIGN_THRESHOLD:
            return "resign", None

        return move_to_usi(best_move), None

    # ----------------------------------------------------------
    # SPSA インターフェース
    # ----------------------------------------------------------

    def get_spsa_params(self) -> Dict[str, float]:
        """現在のパラメータをdictで返す（SPSATunerとの接続用）"""
        return {
            "alpha":       self.alpha,
            "beta":        self.beta,
            "gamma":       self.gamma,
            "delta":       self.delta,
            "epsilon":     self.epsilon,
            "zeta":        self.zeta,
            "w_max":       self.w_max,       # 追加
            "w_mean":      self.w_mean,      # 追加
            "w_count":     self.w_count,     # 追加
            "var_penalty": self.var_penalty,
            "unc_w_var":   self.unc_w_var,
            "unc_w_visit": self.unc_w_visit,
            "unc_w_range": self.unc_w_range,
        }

    def apply_spsa_params(self, params: Dict[str, float]) -> None:
        """SPSATunerから受け取ったパラメータを適用する"""
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, float(val))

    # ----------------------------------------------------------
    # USI制御
    # ----------------------------------------------------------

    def stop(self) -> None:
        self.halt = 0

    def quit(self) -> None:
        self.stop()

    def ponderhit(self, last_limits: dict) -> None:
        self.set_limits(**last_limits)
