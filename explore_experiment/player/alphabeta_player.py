"""αβ探索 + NNUE/ResNet 評価の USI プレイヤー。"""
from __future__ import annotations

import faulthandler
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from cshogi import Board, BLACK, NOT_REPETITION, move_to_usi

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from player.base_player import BasePlayer
from player.evaluator import Evaluator, create_evaluator

faulthandler.enable(all_threads=True)

DEFAULT_GPU_ID = 0
DEFAULT_RESIGN_THRESHOLD = 0.01
DEFAULT_TIME_MARGIN = 1000
DEFAULT_BYOYOMI_MARGIN = 100
DEFAULT_SEARCH_DEPTH = 6
DEFAULT_PV_INTERVAL = 1000
DEFAULT_CONST_NODES = 10_000_000

MATE_SCORE = 1.0
DRAW_SCORE = 0.0


class AlphaBetaPlayer(BasePlayer):
    """negamax + αβ。葉では評価器の勝率（手番側）を [-1,1] に写像して使用。"""

    name = "AlphaBeta-NNUE"
    DEFAULT_MODELFILE = "checkpoint/nnue_default/nnue-002.pth"
    time_limit: Optional[int] = None
    minimum_time: int = 0
    extend_time: bool = False

    def __init__(self, blocks: int = 10) -> None:
        super().__init__()
        self.modelfile: str = self.DEFAULT_MODELFILE
        self.eval_type: str = "nnue"
        self.activation_function: str = "relu"
        self.evaluator: Optional[Evaluator] = None
        self.root_board: Board = Board()
        self.gpu_id: int = DEFAULT_GPU_ID
        self.device: Optional[Union[str, torch.device]] = None
        self.blocks = blocks

        self.resign_threshold: float = DEFAULT_RESIGN_THRESHOLD
        self.time_margin: int = DEFAULT_TIME_MARGIN
        self.byoyomi_margin: int = DEFAULT_BYOYOMI_MARGIN
        self.search_depth: int = DEFAULT_SEARCH_DEPTH
        self.pv_interval: int = DEFAULT_PV_INTERVAL

        self.halt: Optional[int] = None
        self.debug = False

        self._abort = False
        self.nodes_evaluated = 0
        self._last_pv: list[int] = []
        self._last_score: float = 0.0
        self._last_completed_depth: int = 0

    def _log_debug(self, msg: str) -> None:
        if os.environ.get("USI_DEBUG", "").lower() in ("1", "true", "yes"):
            print(f"[alphabeta] {msg}", file=sys.stderr, flush=True)
        if self.debug:
            print(f"info string debug {msg}", flush=True)

    def usi(self) -> None:
        print("id name " + self.name)
        print("option name USI_Ponder type check default false")
        print("option name modelfile type string default " + self.DEFAULT_MODELFILE)
        print("option name gpu_id type spin default " + str(DEFAULT_GPU_ID) + " min -1 max 7")
        print(
            "option name resign_threshold type spin default "
            + str(int(DEFAULT_RESIGN_THRESHOLD * 100))
            + " min 0 max 100"
        )
        print("option name time_margin type spin default " + str(DEFAULT_TIME_MARGIN) + " min 0 max 1000")
        print("option name byoyomi_margin type spin default " + str(DEFAULT_BYOYOMI_MARGIN) + " min 0 max 1000")
        print("option name search_depth type spin default " + str(DEFAULT_SEARCH_DEPTH) + " min 1 max 64")
        print("option name pv_interval type spin default " + str(DEFAULT_PV_INTERVAL) + " min 0 max 10000")
        print(
            "option name activation_function type combo default relu"
            + " var relu var leaky_relu var soft_sin var scaled_arc_tanh var quadratic var cube_root var squash_two"
        )
        print("option name eval_type type combo default nnue var resnet var nnue")
        print("option name debug type check default false")

    def setoption(self, args: list[str]) -> None:
        if args[1] == "modelfile":
            self.modelfile = args[3]
        elif args[1] == "gpu_id":
            self.gpu_id = int(args[3])
        elif args[1] == "resign_threshold":
            self.resign_threshold = int(args[3]) / 100
        elif args[1] == "time_margin":
            self.time_margin = int(args[3])
        elif args[1] == "byoyomi_margin":
            self.byoyomi_margin = int(args[3])
        elif args[1] == "search_depth":
            self.search_depth = int(args[3])
        elif args[1] == "pv_interval":
            self.pv_interval = int(args[3])
        elif args[1] == "activation_function":
            self.activation_function = args[3]
        elif args[1] == "eval_type":
            self.eval_type = args[3]
        elif args[1] == "debug":
            self.debug = args[3] == "true"

    def load_evaluator(self) -> None:
        if self.device is None:
            raise RuntimeError("device not set")
        self.evaluator = create_evaluator(
            eval_type=self.eval_type,
            device=self.device,
            modelfile=self.modelfile,
            blocks=self.blocks,
            activation_function=self.activation_function,
        )
        self.evaluator.warmup()

    def isready(self) -> None:
        self._log_debug(
            f"isready eval_type={self.eval_type} modelfile={self.modelfile} gpu_id={self.gpu_id}"
        )
        if self.gpu_id >= 0:
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")
        try:
            self.load_evaluator()
        except Exception as e:
            self._log_debug(f"load_evaluator failed: {e}")
            raise
        self.root_board.reset()
        self._log_debug(f"isready ok device={self.device}")

    def position(self, sfen: str, usi_moves: list[str]) -> None:
        if sfen == "startpos":
            self.root_board.reset()
        elif sfen[:5] == "sfen ":
            self.root_board.set_sfen(sfen[5:])
        for usi_move in usi_moves:
            self.root_board.push_usi(usi_move)
        if self.debug:
            print(self.root_board)

    def set_limits(
        self,
        btime: Optional[int] = None,
        wtime: Optional[int] = None,
        byoyomi: Optional[int] = None,
        binc: Optional[int] = None,
        winc: Optional[int] = None,
        nodes: Optional[int] = None,
        infinite: bool = False,
        ponder: bool = False,
    ) -> None:
        if infinite or ponder:
            self.halt = 2**31 - 1
            self.time_limit = None
        elif nodes:
            self.halt = nodes
            self.time_limit = None
        else:
            self.minimum_time = 0
            self.remaining_time, inc = (btime, binc) if self.root_board.turn == BLACK else (wtime, winc)
            if self.remaining_time is None and byoyomi is None and inc is None:
                self.halt = DEFAULT_CONST_NODES
                self.time_limit = None
            else:
                self.remaining_time = int(self.remaining_time) if self.remaining_time else 0
                inc = int(inc) if inc else 0
                self.time_limit = self.remaining_time / (14 + max(0, 30 - self.root_board.move_number)) + inc
                if byoyomi:
                    byoyomi = int(byoyomi) - self.byoyomi_margin
                    self.minimum_time = byoyomi
                    if self.time_limit < byoyomi:
                        self.time_limit = byoyomi
                self.extend_time = self.time_limit > self.minimum_time
                self.halt = None

    def stop(self) -> None:
        self.halt = 0
        self._abort = True

    def ponderhit(self, last_limits: dict) -> None:
        self.begin_time = time.time()
        self.set_limits(**last_limits)

    def quit(self) -> None:
        self.stop()

    def _time_up(self) -> bool:
        if self.halt is not None and self.nodes_evaluated >= self.halt:
            return True
        if getattr(self, "time_limit", None) is None:
            return False
        spend_ms = int((time.time() - self.begin_time) * 1000)
        if spend_ms * 10 < self.time_limit:
            return False
        if spend_ms < getattr(self, "minimum_time", 0):
            return False
        return spend_ms >= self.time_limit

    def _leaf_eval_nn(self, board: Board) -> float:
        """手番側勝率 p を [-1,1] のスコアに（葉・手番視点）。"""
        if self.evaluator is None:
            return 0.0
        legal = list(board.legal_moves)
        _, vals = self.evaluator.evaluate_batch([board], [legal], [board.turn], 1.0)
        self.nodes_evaluated += 1
        p = float(np.asarray(vals[0]).reshape(-1)[0])
        return 2.0 * p - 1.0

    def _leaf_eval(self, board: Board) -> float:
        if board.is_draw() != NOT_REPETITION:
            # 千日手系は葉では引き分け近似（詳細な勝敗フラグは将来拡張）
            return DRAW_SCORE
        return self._leaf_eval_nn(board)

    def _negamax(
        self, board: Board, depth: int, alpha: float, beta: float
    ) -> tuple[float, list[int]]:
        if self._abort or self._time_up():
            return float("nan"), []

        legal = list(board.legal_moves)
        if not legal:
            if board.is_check():
                return -MATE_SCORE, []
            return DRAW_SCORE, []

        if depth == 0:
            if self._abort or self._time_up():
                return float("nan"), []
            return self._leaf_eval(board), []

        best_score = -float("inf")
        best_pv: list[int] = []

        for move in legal:
            if self._abort or self._time_up():
                return float("nan"), []
            board.push(move)
            score, child_pv = self._negamax(board, depth - 1, -beta, -alpha)
            if math.isnan(score):
                board.pop()
                return float("nan"), []
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_pv = [move] + child_pv

            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if math.isnan(best_score) or best_score == -float("inf"):
            return float("nan"), []
        return best_score, best_pv

    def _iterative_search(self, board: Board) -> tuple[float, list[int], int]:
        self._abort = False
        self.nodes_evaluated = 0
        best_pv: list[int] = []
        best_score = 0.0
        last_d = 0
        last_info_ms = 0

        # search_depth は「最低限ここまでは読む目安」。
        # 実際の打ち切りは MCTS/NPLS と同様に time/nodes/halt に委ねる。
        max_d = 64
        for d in range(1, max_d + 1):
            if self._abort or self._time_up():
                break
            sc, pv = self._negamax_root(board, d)
            if pv and not math.isnan(sc):
                best_score = sc
                best_pv = pv
                last_d = d
                self._last_score = best_score
                self._last_pv = list(best_pv)
                self._last_completed_depth = last_d
                if self.pv_interval > 0:
                    elapsed_ms = int((time.time() - self.begin_time) * 1000)
                    if elapsed_ms >= last_info_ms + self.pv_interval:
                        last_info_ms = elapsed_ms
                        self._print_info(time.time() - self.begin_time)
            # 目安深さ到達後は、時間/ノード条件を満たした時点で終了
            if d >= self.search_depth and (self._abort or self._time_up()):
                break

        return best_score, best_pv, last_d

    def _negamax_root(self, board: Board, depth: int) -> tuple[float, list[int]]:
        legal = list(board.legal_moves)
        if not legal:
            return 0.0, []
        if len(legal) == 1:
            return 0.0, [legal[0]]

        best_score = -float("inf")
        best_pv: list[int] = []

        alpha = -float("inf")
        beta = float("inf")

        for move in legal:
            if self._abort or self._time_up():
                break
            board.push(move)
            score, child_pv = self._negamax(board, depth - 1, -beta, -alpha)
            if math.isnan(score):
                board.pop()
                break
            score = -score
            board.pop()

            if score > best_score:
                best_score = score
                best_pv = [move] + child_pv

            if score > alpha:
                alpha = score

        if math.isnan(best_score) or best_score == -float("inf"):
            return float("nan"), []
        return best_score, best_pv

    def _score_to_cp(self, score: float) -> int:
        if math.isnan(score):
            return 0
        p = (score + 1.0) / 2.0
        p = max(min(p, 1.0 - 1e-9), 1e-9)
        if p >= 1.0:
            return 30000
        if p <= 0.0:
            return -30000
        return int(-math.log(1.0 / p - 1.0) * 600)

    def _print_info(self, finish_time: float) -> None:
        pv_s = " ".join(move_to_usi(m) for m in self._last_pv) if self._last_pv else ""
        cp = self._score_to_cp(self._last_score)
        print(
            "info depth {} score cp {} nodes {} time {} pv {}".format(
                self._last_completed_depth,
                cp,
                self.nodes_evaluated,
                int(finish_time * 1000),
                pv_s,
            ),
            flush=True,
        )

    def go(self) -> tuple[str, Optional[str]]:
        self.begin_time = time.time()
        self._abort = False

        if self.root_board.is_game_over():
            return "resign", None
        if self.root_board.is_nyugyoku():
            return "win", None

        if not self.root_board.is_check():
            m1 = self.root_board.mate_move_in_1ply()
            if m1:
                print("info score mate 1 pv {}".format(move_to_usi(m1)), flush=True)
                return move_to_usi(m1), None

        m3 = self.root_board.mate_move(3)
        if m3 != 0:
            print("info score mate 3 pv {}".format(move_to_usi(m3)), flush=True)
            return move_to_usi(m3), None

        legal = list(self.root_board.legal_moves)
        if len(legal) == 1:
            return move_to_usi(legal[0]), None

        if self.evaluator is None:
            print("info string error evaluator not loaded", flush=True)
            return move_to_usi(legal[0]), None

        b = self.root_board.copy()
        self._last_score, self._last_pv, self._last_completed_depth = self._iterative_search(b)

        finish_time = time.time() - self.begin_time
        self._print_info(finish_time)

        if not self._last_pv:
            return move_to_usi(legal[0]), None

        best_usi = move_to_usi(self._last_pv[0])
        ponder_usi: Optional[str] = None
        if len(self._last_pv) >= 2:
            ponder_usi = move_to_usi(self._last_pv[1])

        p_root = (self._last_score + 1.0) / 2.0
        if math.isnan(self._last_score):
            p_root = 0.5
        if p_root < self.resign_threshold:
            return "resign", None

        return best_usi, ponder_usi


if __name__ == "__main__":
    AlphaBetaPlayer().run()
