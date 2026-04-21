"""
search/ads_ab.py
Adaptive Depth Search Alpha-Beta (ADS-AB) の実装。
局面の「危機度」に応じて探索深さを動的に変更する。
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
DEFAULT_RESIGN_THRESHOLD = 0.05
DEFAULT_BLOCKS_CONFIG    = "30blocks"
MAX_DEPTH                = 20
VALUE_WIN                = 1.0
VALUE_LOSE               = 0.0

class ADSABPlayer:
    """
    USI対局エンジン。
    Adaptive Depth Search Alpha-Beta プレイヤー。
    """
    name = "GNN-Shogi-ADSAB"
    DEFAULT_MODELFILE = "weights/checkpoint.pth"

    def __init__(
        self,
        features_mode: int = 0,
        blocks_config_mode: str = DEFAULT_BLOCKS_CONFIG,
        name: str = "GNN-Shogi-ADSAB",
        modelfile: str = "weights/checkpoint.pth",
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
        self.resign_threshold = DEFAULT_RESIGN_THRESHOLD
        
        self.halt = None
        self.time_limit = 0
        self.minimum_time = 0
        self.nodes_searched = 0

    def usi(self) -> None:
        print(f"id name {self.name}")
        print(f"option name modelfile type string default {self.DEFAULT_MODELFILE}")
        print(f"option name gpu_id type spin default {DEFAULT_GPU_ID} min -1 max 7")
        print("option name debug type check default false")

    def setoption(self, args: list) -> None:
        if args[1] == "modelfile":
            self.modelfile = args[3]
        elif args[1] == "gpu_id":
            self.gpu_id = int(args[3])
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
                self.halt = 1000 # Default nodes
            else:
                remaining_time = int(remaining_time) if remaining_time else 0
                inc = int(inc) if inc else 0
                self.time_limit = remaining_time / 30 + inc
                self.minimum_time = int(byoyomi) - 100 if byoyomi else 0
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
            
            # 補助タスクから「危機度 (Crisis Level)」を算出
            # 敵の王への攻撃 (threat) や自陣のダメージ (damage) を考慮
            threat = aux['threat'].mean().item()
            damage = aux['damage'].mean().item()
            crisis_level = max(threat, damage)
            
        return policy, value, crisis_level

    def alpha_beta(self, board: Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        self.nodes_searched += 1
        
        if board.is_game_over():
            return VALUE_LOSE if is_maximizing else VALUE_WIN
        
        if depth <= 0:
            _, value, _ = self.evaluate(board)
            return value

        policy, _, crisis = self.evaluate(board)
        
        # 危機度が高い場合、探索深さを延長 (ADS)
        if crisis > 0.7 and depth == 1:
            depth += 1

        legal_moves = list(board.legal_moves)
        # Policyに基づいて指し手をソート（枝刈り効率化）
        move_scores = []
        for m in legal_moves:
            label = make_move_label(m, board.turn)
            move_scores.append((policy[label], m))
        move_scores.sort(key=lambda x: x[0], reverse=True)

        if is_maximizing:
            max_eval = -float('inf')
            for _, move in move_scores:
                board.push(move)
                eval = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
                if self.check_halt(): break
            return max_eval
        else:
            min_eval = float('inf')
            for _, move in move_scores:
                board.push(move)
                eval = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
                if self.check_halt(): break
            return min_eval

    def check_halt(self) -> bool:
        elapsed = (time.time() - self.begin_time) * 1000
        if self.halt and self.nodes_searched >= self.halt: return True
        if not self.halt and elapsed >= self.time_limit: return True
        return False

    def go(self) -> tuple:
        self.begin_time = time.time()
        self.nodes_searched = 0
        best_move = None
        
        # 反復深化
        for depth in range(1, MAX_DEPTH):
            self.nodes_searched = 0
            legal_moves = list(self.root_board.legal_moves)
            policy, _, _ = self.evaluate(self.root_board)
            
            move_scores = []
            for m in legal_moves:
                label = make_move_label(m, self.root_board.turn)
                move_scores.append((policy[label], m))
            move_scores.sort(key=lambda x: x[0], reverse=True)
            
            current_best_move = None
            max_val = -float('inf')
            
            for _, move in move_scores:
                self.root_board.push(move)
                val = self.alpha_beta(self.root_board, depth - 1, -float('inf'), float('inf'), False)
                self.root_board.pop()
                
                if val > max_val:
                    max_val = val
                    current_best_move = move
                
                if self.check_halt(): break
            
            if current_best_move:
                best_move = current_best_move
            
            elapsed = (time.time() - self.begin_time) * 1000
            cp = int(-math.log(1.0 / max(1e-4, min(1-1e-4, max_val)) - 1.0) * 600)
            print(f"info depth {depth} nodes {self.nodes_searched} time {int(elapsed)} score cp {cp} pv {move_to_usi(best_move)}", flush=True)
            
            if self.check_halt(): break

        return move_to_usi(best_move), None

    def stop(self) -> None:
        self.halt = 0

    def quit(self) -> None:
        self.stop()

    def ponderhit(self, last_limits: dict) -> None:
        self.begin_time = time.time()
        self.set_limits(**last_limits)