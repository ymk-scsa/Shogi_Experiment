"""
search/mc_rzf.py
Monte Carlo Relational Zero Search (MC-RZF) の実装。
MCTSPlayer と同等のインターフェースを持ちつつ、関係性（Relational）に基づいた探索制御を行う。
"""

import numpy as np
import torch
import torch.nn.functional as F
import time
import math
import sys
import os
from typing import Optional, Union

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

# ===== 定数 =====
DEFAULT_GPU_ID           = 0
DEFAULT_BATCH_SIZE       = 16
DEFAULT_RESIGN_THRESHOLD = 0.05
DEFAULT_C_PUCT           = 1.5
DEFAULT_TEMPERATURE      = 1.0
DEFAULT_TIME_MARGIN      = 1000
DEFAULT_BYOYOMI_MARGIN   = 100
DEFAULT_PV_INTERVAL      = 500
DEFAULT_BLOCKS_CONFIG    = "30blocks"

# MCRZF 固有: 関係性重み
RELATIONAL_ALPHA = 0.5

class MCRZFNode:
    def __init__(self) -> None:
        self.move_count: int = 0
        self.sum_value: float = 0.0
        self.child_move: Optional[list] = None
        self.child_move_count: Optional[np.ndarray] = None
        self.child_sum_value: Optional[np.ndarray] = None
        self.child_node: Optional[list] = None
        self.policy: Optional[np.ndarray] = None
        self.value: Optional[float] = None
        # 関係性に基づく補正ベクトル (将来的に CAMoRN 等から取得)
        self.relational_prior: Optional[np.ndarray] = None

    def expand(self, board: Board) -> None:
        self.child_move = list(board.legal_moves)
        child_num = len(self.child_move)
        self.child_move_count = np.zeros(child_num, dtype=np.int32)
        self.child_sum_value = np.zeros(child_num, dtype=np.float32)
        self.child_node = [None] * child_num

class MCRZFPlayer:
    """
    USI対局エンジン。
    Monte Carlo Relational Zero Search プレイヤー。
    """
    name = "GNN-Shogi-MCRZF"
    DEFAULT_MODELFILE = "weights/checkpoint.pth"

    def __init__(
        self,
        features_mode: int = 0,
        blocks_config_mode: str = DEFAULT_BLOCKS_CONFIG,
        name: str = "GNN-Shogi-MCRZF",
        modelfile: str = "weights/checkpoint.pth",
    ) -> None:
        self.name = name
        self.modelfile = modelfile
        self.model = None
        self.device = None
        self.gpu_id = DEFAULT_GPU_ID
        self.batch_size = DEFAULT_BATCH_SIZE
        
        self.features_setting = FEATURES_SETTINGS[features_mode]
        self.blocks_config_mode = blocks_config_mode
        self.debug = False

        self.root_board = Board()
        self.root_node = MCRZFNode()
        
        self.c_puct = DEFAULT_C_PUCT
        self.temperature = DEFAULT_TEMPERATURE
        self.resign_threshold = DEFAULT_RESIGN_THRESHOLD
        
        self.halt = None
        self.time_limit = 0
        self.minimum_time = 0

    def usi(self) -> None:
        print(f"id name {self.name}")
        print(f"option name modelfile type string default {self.DEFAULT_MODELFILE}")
        print(f"option name gpu_id type spin default {DEFAULT_GPU_ID} min -1 max 7")
        print(f"option name batchsize type spin default {DEFAULT_BATCH_SIZE} min 1 max 256")
        print(f"option name c_puct type spin default {int(self.c_puct*100)} min 10 max 1000")
        print("option name debug type check default false")

    def setoption(self, args: list) -> None:
        if args[1] == "modelfile":
            self.modelfile = args[3]
        elif args[1] == "gpu_id":
            self.gpu_id = int(args[3])
        elif args[1] == "batchsize":
            self.batch_size = int(args[3])
        elif args[1] == "c_puct":
            self.c_puct = int(args[3]) / 100
        elif args[1] == "debug":
            self.debug = args[3] == "true"

    def isready(self) -> None:
        self.device = torch.device(f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu")
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
        self.root_node = MCRZFNode()

    def position(self, sfen: str, usi_moves: list) -> None:
        if sfen == "startpos":
            self.root_board.reset()
        elif sfen.startswith("sfen "):
            self.root_board.set_sfen(sfen[5:])
        
        for usi_move in usi_moves:
            self.root_board.push_usi(usi_move)
        
        # 簡易的なツリーリセット（前回の探索結果を引き継がない）
        self.root_node = MCRZFNode()

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
                self.halt = 1000
            else:
                remaining_time = int(remaining_time) if remaining_time else 0
                inc = int(inc) if inc else 0
                self.time_limit = remaining_time / 20 + inc
                self.minimum_time = int(byoyomi) - DEFAULT_BYOYOMI_MARGIN if byoyomi else 0
                if self.time_limit < self.minimum_time:
                    self.time_limit = self.minimum_time
                self.halt = None

    def evaluate(self, board: Board) -> tuple:
        features = torch.zeros((1, self.features_setting.features_num, 9, 9), dtype=torch.float32).to(self.device)
        self.features_setting.make_features(board, features[0].cpu().numpy())
        
        with torch.no_grad():
            policy_logits, value_tanh, aux = self.model(features, return_aux=True)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = (value_tanh.item() + 1.0) / 2.0
            
            # Relational Prior として Search Rep (探索表現) や Future Rep (未来表現) を活用
            # ここでは将来的な CAMoRN 統合のプレースホルダーとして policy を補正する。
            # 例: search_rep の L2 ノルムが高い指し手ほど優先するなど
            rel_weight = torch.norm(aux['search_rep'], p=2, dim=1).item()
            # 簡易的に policy を少し尖らせる
            policy = policy ** (1.0 + 0.1 * rel_weight)
            policy /= policy.sum()
            
        return policy, value

    def select_child(self, node: MCRZFNode) -> int:
        # PUCT スコア計算
        q = np.divide(
            node.child_sum_value,
            node.child_move_count,
            out=np.zeros(len(node.child_move), np.float32),
            where=node.child_move_count != 0
        )
        u = self.c_puct * node.policy * (math.sqrt(node.move_count) / (1 + node.child_move_count))
        score = q + u
        return int(np.argmax(score))

    def search(self, board: Board, node: MCRZFNode) -> float:
        if node.value is not None and (node.value == 0.0 or node.value == 1.0):
            return 1.0 - node.value

        if node.child_move is None:
            # 展開と評価
            policy_vec, value = self.evaluate(board)
            node.expand(board)
            
            legal_policy = np.zeros(len(node.child_move), dtype=np.float32)
            for i, move in enumerate(node.child_move):
                label = make_move_label(move, board.turn)
                legal_policy[i] = policy_vec[label]
            
            # ソフトマックス正規化
            if legal_policy.sum() > 0:
                legal_policy /= legal_policy.sum()
            else:
                legal_policy = np.ones(len(node.child_move)) / len(node.child_move)
            
            node.policy = legal_policy
            node.value = value
            return 1.0 - value

        # 選択
        idx = self.select_child(node)
        move = node.child_move[idx]
        
        board.push(move)
        if node.child_node[idx] is None:
            node.child_node[idx] = MCRZFNode()
        
        v = self.search(board, node.child_node[idx])
        board.pop()
        
        # 更新
        node.child_move_count[idx] += 1
        node.child_sum_value[idx] += v
        node.move_count += 1
        node.sum_value += v
        
        return 1.0 - v

    def go(self) -> tuple:
        self.begin_time = time.time()
        playouts = 0
        
        if self.root_node.child_move is None:
            self.root_node.expand(self.root_board)
            policy, value = self.evaluate(self.root_board)
            legal_policy = np.zeros(len(self.root_node.child_move), dtype=np.float32)
            for i, move in enumerate(self.root_node.child_move):
                label = make_move_label(move, self.root_board.turn)
                legal_policy[i] = policy[label]
            if legal_policy.sum() > 0: legal_policy /= legal_policy.sum()
            self.root_node.policy = legal_policy
            self.root_node.value = value

        while True:
            elapsed = (time.time() - self.begin_time) * 1000
            if self.halt and playouts >= self.halt: break
            if not self.halt and elapsed >= self.time_limit: break
            
            board_copy = self.root_board.copy()
            self.search(board_copy, self.root_node)
            playouts += 1
            
            if playouts % 100 == 0:
                # PV情報の表示
                best_idx = np.argmax(self.root_node.child_move_count)
                best_move = self.root_node.child_move[best_idx]
                score = (self.root_node.child_sum_value[best_idx] / self.root_node.child_move_count[best_idx])
                cp = int(-math.log(1.0 / max(1e-4, min(1-1e-4, score)) - 1.0) * 600)
                print(f"info nodes {playouts} time {int(elapsed)} score cp {cp} pv {move_to_usi(best_move)}", flush=True)

        best_idx = np.argmax(self.root_node.child_move_count)
        best_move = self.root_node.child_move[best_idx]
        
        # 簡易的に ponder は None
        return move_to_usi(best_move), None

    def stop(self) -> None:
        self.halt = 0

    def quit(self) -> None:
        self.stop()

    def ponderhit(self, last_limits: dict) -> None:
        self.begin_time = time.time()
        self.set_limits(**last_limits)