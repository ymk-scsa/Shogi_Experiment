"""
search/npls.py
Neural Priority List Search (NPLS) の実装。
USIエンジンとして動作するように MCTSPlayer と同等のインターフェースを持つ NPLSPlayer を実装。
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
import math
import sys
import os
import heapq
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Tuple
from collections import defaultdict

from cshogi import (
    Board,
    BLACK,
    NOT_REPETITION,
    REPETITION_DRAW,
    REPETITION_WIN,
    REPETITION_SUPERIOR,
    move_to_usi,
)

# プロジェクトルートを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from game.board import (
    FEATURES_SETTINGS,
    MOVE_LABELS_NUM,
    make_move_label,
)
from model.model import create_model

# ===== 定数 (mcts.py と合わせる) =====
DEFAULT_GPU_ID           = 0
DEFAULT_BATCH_SIZE       = 1  # NPLSは一旦バッチサイズ1で実装（シンプルさ優先）
DEFAULT_RESIGN_THRESHOLD = 0.01
DEFAULT_TIME_MARGIN      = 1000
DEFAULT_BYOYOMI_MARGIN   = 100
DEFAULT_PV_INTERVAL      = 500
DEFAULT_CONST_NODES      = 1000
DEFAULT_BLOCKS_CONFIG    = "30blocks"

# NPLS 固有の重みパラメータ
DEFAULT_ALPHA   = 1.0  # Q (Value)
DEFAULT_BETA    = 0.5  # Policy
DEFAULT_GAMMA   = 0.2  # Depth
DEFAULT_DELTA   = 0.3  # Uncertainty (Entropy)
DEFAULT_EPSILON = 0.5  # Exploration (1/sqrt(N))
# 追加（NPLSパラメータの下）UW-NPLS
DEFAULT_ZETA = 0.7  # Uncertainty bonus (NEW)
DEFAULT_VAR_PENALTY = 0.3  # Root variance penalty (NEW)

@dataclass(order=True)
class NPLSNode:
    priority: float
    state_sfen: str = field(compare=False)
    first_move: Optional[int] = field(compare=False)
    value: float = field(compare=False)
    policy: float = field(compare=False)
    depth: int = field(compare=False)
    uncertainty: float = field(compare=False)
    visit_count: int = field(default=1, compare=False)
    total_value: float = field(default=0.0, compare=False)

    def update(self, new_value: float):
        self.visit_count += 1
        self.total_value += new_value
        self.value = self.total_value / self.visit_count

class RootMoveStats:
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "max_value": -float("inf"),
            "sum_value": 0.0,
            "sum_sq": 0.0,  # NEW
            "count": 0
        })

    def update(self, move, value):
        s = self.stats[move]
        s["max_value"] = max(s["max_value"], value)
        s["sum_value"] += value
        s["sum_sq"] += value * value  # NEW
        s["count"] += 1

    def score(self, move,
              w_max=0.6,
              w_mean=0.3,
              w_count=0.1,
              var_penalty=0.3):

        s = self.stats[move]
        if s["count"] == 0:
            return -float("inf")

        mean = s["sum_value"] / s["count"]

        variance = max(
            (s["sum_sq"] / s["count"]) - mean * mean,
            0.0
        )

        return (
            w_max * s["max_value"]
            + w_mean * mean
            + w_count * math.log(s["count"] + 1)
            - var_penalty * variance  # NEW
        )

    def best_move(self, var_penalty=0.3):
        if not self.stats:
            return None
        return max(
            self.stats.keys(),
            key=lambda m: self.score(m, var_penalty=var_penalty)
        )

class NPLSPlayer:
    """
    USI対局エンジン。
    Neural Priority List Search を用いたプレイヤー。
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
        self.name = name
        self.modelfile = modelfile
        self.model = None
        self.device = None
        self.gpu_id = DEFAULT_GPU_ID
        
        self.features_setting = FEATURES_SETTINGS[features_mode]
        self.blocks_config_mode = blocks_config_mode
        self.debug = False

        self.root_board = Board()
        self.halt = None
        self.time_limit = 0
        self.minimum_time = 0
        
        # NPLS パラメータ
        self.alpha = DEFAULT_ALPHA
        self.beta = DEFAULT_BETA
        self.gamma = DEFAULT_GAMMA
        self.delta = DEFAULT_DELTA
        self.epsilon = DEFAULT_EPSILON
        self.zeta = DEFAULT_ZETA
        self.var_penalty = DEFAULT_VAR_PENALTY

        self.transposition_table = {}
        self.open_list = []
        self.root_stats = RootMoveStats()
        self.nodes_searched = 0

    def usi(self) -> None:
        print(f"id name {self.name}")
        print(f"option name modelfile type string default {self.DEFAULT_MODELFILE}")
        print(f"option name gpu_id type spin default {DEFAULT_GPU_ID} min -1 max 7")
        print(f"option name alpha type spin default {int(self.alpha*100)} min 0 max 1000")
        print(f"option name beta type spin default {int(self.beta*100)} min 0 max 1000")
        print(f"option name gamma type spin default {int(self.gamma*100)} min 0 max 1000")
        print(f"option name delta type spin default {int(self.delta*100)} min 0 max 1000")
        print(f"option name epsilon type spin default {int(self.epsilon*100)} min 0 max 1000")
        print(f"option name zeta type spin default {int(self.zeta*100)} min 0 max 1000")
        print(f"option name var_penalty type spin default {int(self.var_penalty*100)} min 0 max 1000")
        print("option name debug type check default false")

    def setoption(self, args: list) -> None:
        if args[1] == "modelfile":
            self.modelfile = args[3]
        elif args[1] == "gpu_id":
            self.gpu_id = int(args[3])
        elif args[1] == "alpha":
            self.alpha = int(args[3]) / 100
        elif args[1] == "beta":
            self.beta = int(args[3]) / 100
        elif args[1] == "gamma":
            self.gamma = int(args[3]) / 100
        elif args[1] == "delta":
            self.delta = int(args[3]) / 100
        elif args[1] == "epsilon":
            self.epsilon = int(args[3]) / 100
        elif args[1] == "zeta":
            self.zeta = int(args[3]) / 100
        elif args[1] == "var_penalty":
            self.var_penalty = int(args[3]) / 100  
        elif args[1] == "debug":
            self.debug = args[3] == "true"

    def isready(self) -> None:
        if self.gpu_id >= 0:
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")
        
        self.model = create_model(
            input_channels=self.features_setting.features_num,
            num_actions=MOVE_LABELS_NUM,
            mode=self.blocks_config_mode
        )
        self.model.to(self.device)
        if os.path.exists(self.modelfile):
            checkpoint = torch.load(self.modelfile, map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.root_board.reset()

    def position(self, sfen: str, usi_moves: list) -> None:
        if sfen == "startpos":
            self.root_board.reset()
        elif sfen.startswith("sfen "):
            self.root_board.set_sfen(sfen[5:])
        
        for usi_move in usi_moves:
            self.root_board.push_usi(usi_move)

    def set_limits(self, **kwargs) -> None:
        if kwargs.get("infinite") or kwargs.get("ponder"):
            self.halt = 10**9
        elif kwargs.get("nodes"):
            self.halt = kwargs["nodes"]
        else:
            btime = kwargs.get("btime")
            wtime = kwargs.get("wtime")
            byoyomi = kwargs.get("byoyomi")
            inc = kwargs.get("binc") if self.root_board.turn == BLACK else kwargs.get("winc")
            remaining_time = btime if self.root_board.turn == BLACK else wtime
            
            if remaining_time is None and byoyomi is None:
                self.halt = DEFAULT_CONST_NODES
            else:
                remaining_time = int(remaining_time) if remaining_time else 0
                inc = int(inc) if inc else 0
                self.time_limit = remaining_time / 20 + inc # 簡易的な時間管理
                self.minimum_time = int(byoyomi) - DEFAULT_BYOYOMI_MARGIN if byoyomi else 0
                if self.time_limit < self.minimum_time:
                    self.time_limit = self.minimum_time
                self.halt = None

    def compute_priority(self, node: NPLSNode):
        Q = node.value
        P = node.policy
        D = math.log(node.depth + 1)
        U = node.uncertainty
        E = 1.0 / math.sqrt(node.visit_count + 1)

        # ===== UW-NPLS 拡張 =====
        UW = U / (1.0 + math.sqrt(node.visit_count))

        return (
            self.alpha * Q
            + self.beta * P
            + self.gamma * D
            + self.delta * U
            + self.epsilon * E
            + self.zeta * UW   # NEW
        )

    def push_node(self, node: NPLSNode):
        node.priority = self.compute_priority(node)
        heapq.heappush(self.open_list, (-node.priority, node))

    def evaluate(self, board: Board) -> Tuple[float, np.ndarray, float]:
        """モデルを呼び出して価値、方策、不確実性を取得"""
        features = torch.zeros(
            (1, self.features_setting.features_num, 9, 9),
            dtype=torch.float32
        ).to(self.device)
        
        self.features_setting.make_features(board, features[0].cpu().numpy())
        
        with torch.no_grad():
            policy_logits, value_tanh, aux = self.model(features, return_aux=True)
            
            # policy: softmax
            policy = torch.exp(F.log_softmax(policy_logits, dim=1)).cpu().numpy()[0]
            # value: Tanh -> [0, 1]
            value = ((value_tanh.item() + 1.0) / 2.0)
            # uncertainty: policy_entropy (aux から取得)
            uncertainty = aux['policy_entropy'].item()
            
        return value, policy, uncertainty

    def go(self) -> Tuple[str, Optional[str]]:
        self.begin_time = time.time()
        self.nodes_searched = 0
        self.open_list = []
        self.transposition_table = {}
        self.root_stats = RootMoveStats()

        if self.root_board.is_game_over():
            return "resign", None

        # Root evaluation
        root_value, root_policy, root_uncertainty = self.evaluate(self.root_board)
        root_node = NPLSNode(
            priority=0.0,
            state_sfen=self.root_board.sfen(),
            first_move=None,
            value=root_value,
            policy=1.0, # Root policy is not used for priority
            depth=0,
            uncertainty=root_uncertainty,
            total_value=root_value
        )
        self.push_node(root_node)

        while True:
            if not self.open_list:
                break
            
            # 終了判定
            elapsed = (time.time() - self.begin_time) * 1000
            if self.halt and self.nodes_searched >= self.halt:
                break
            if not self.halt and elapsed >= self.time_limit:
                break

            # Pop best node
            _, node = heapq.heappop(self.open_list)
            self.nodes_searched += 1

            # Expand
            board = Board(node.state_sfen)
            if board.is_game_over():
                continue

            for move in board.legal_moves:
                board.push(move)
                child_sfen = board.sfen()
                
                if child_sfen in self.transposition_table:
                    existing = self.transposition_table[child_sfen]
                    existing.update(existing.value)
                    self.push_node(existing)
                    board.pop()
                    continue

                val, pol_vec, unc = self.evaluate(board)
                move_label = make_move_label(move, board.turn ^ 1) # push後のturnの逆
                move_policy = pol_vec[move_label]

                child = NPLSNode(
                    priority=0.0,
                    state_sfen=child_sfen,
                    first_move=node.first_move if node.depth > 0 else move,
                    value=val,
                    policy=move_policy,
                    depth=node.depth + 1,
                    uncertainty=unc,
                    total_value=val
                )
                self.transposition_table[child_sfen] = child
                self.root_stats.update(child.first_move, val)
                self.push_node(child)
                board.pop()

            if self.nodes_searched % 100 == 0 and self.debug:
                print(f"info nodes {self.nodes_searched} time {int(elapsed)} score cp {int(root_value*1000)}")

        best_move = self.root_stats.best_move(self.var_penalty)
        if best_move is None:
            # 万が一探索に失敗した場合は合法手から適当に選ぶ
            best_move = list(self.root_board.legal_moves)[0]

        return move_to_usi(best_move), None

    def stop(self) -> None:
        self.halt = 0

    def quit(self) -> None:
        self.stop()
    
    def ponderhit(self, last_limits: dict) -> None:
        # NPLSでのponderは現状簡易実装
        self.begin_time = time.time()
        self.set_limits(**last_limits)