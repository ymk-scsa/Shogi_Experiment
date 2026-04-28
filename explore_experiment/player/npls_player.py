import faulthandler
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math
import time
from typing import Optional, Union

import numpy as np
import torch
from cshogi import Board, BLACK, move_to_usi
from player.base_player import BasePlayer
from player.evaluator import Evaluator, create_evaluator
from player.npls_node import NPLSNode, NPLSNodeTree
from shogi.feature import make_move_label

faulthandler.enable(all_threads=True)

DEFAULT_GPU_ID = 0
DEFAULT_BATCH_SIZE = 32
DEFAULT_RESIGN_THRESHOLD = 0.01
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TIME_MARGIN = 1000
DEFAULT_BYOYOMI_MARGIN = 100
DEFAULT_PV_INTERVAL = 500
DEFAULT_CONST_PLAYOUT = 1000
DEFAULT_ACTIVATION_FUNCTION = "relu"


def softmax_temperature_with_normalize(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits.astype(np.float64) / temperature
    max_logit = float(np.max(logits))
    probabilities = np.exp(logits - max_logit)
    probabilities /= np.sum(probabilities)
    return probabilities.astype(np.float32)


def value_from_root_perspective(board: Board, root_turn: int, side_to_move_win_prob: float) -> float:
    """NNが出す『手番側勝率』を、探索開始時の手番（ルート）から見た勝率に変換する。"""
    if board.turn == root_turn:
        return side_to_move_win_prob
    return 1.0 - side_to_move_win_prob


def total_value_to_mean_win_prob(total_value: float, depth: int) -> float:
    """total_value は各手順で [0,1] の値を足しただけなので、深さが進むと 1 を大きく超える。
    USI の score cp や投了閾値向けには、深さ平均でおおむね [0,1] に収める。
    """
    d = max(1, depth)
    return float(min(1.0, max(0.0, total_value / d)))


class NPLSPlayer(BasePlayer):
    name = "NPLS-ResNet"
    DEFAULT_MODELFILE = "checkpoint/resnet_10/cache/checkpoint-007.pth"
    time_limit: Optional[int] = None
    minimum_time: int = 0

    def __init__(self, blocks: int = 10) -> None:
        super().__init__()
        self.modelfile: str = self.DEFAULT_MODELFILE
        self.eval_type: str = "resnet"
        self.evaluator: Optional[Evaluator] = None
        self.nodes: list[NPLSNode] = []
        self.current_batch_index: int = 0

        self.root_board: Board = Board()
        self.tree: NPLSNodeTree = NPLSNodeTree()

        self.playout_count: int = 0
        self.halt: Optional[int] = None

        self.gpu_id: int = DEFAULT_GPU_ID
        self.device: Optional[Union[str, torch.device, int]] = None
        self.batch_size: int = DEFAULT_BATCH_SIZE

        self.resign_threshold: float = DEFAULT_RESIGN_THRESHOLD
        self.temperature: float = DEFAULT_TEMPERATURE
        self.time_margin: int = DEFAULT_TIME_MARGIN
        self.byoyomi_margin: int = DEFAULT_BYOYOMI_MARGIN
        self.pv_interval: int = DEFAULT_PV_INTERVAL

        self.blocks = blocks
        self.activation_function: str = DEFAULT_ACTIVATION_FUNCTION
        self.debug = False

        self._best_value: float = 0.0
        self._best_node: Optional[NPLSNode] = None

    def _log_debug(self, msg: str) -> None:
        """USI の stdout を汚さず stderr / info string に出す。環境変数 USI_DEBUG=1 または setoption debug true で有効。"""
        if os.environ.get("USI_DEBUG", "").lower() in ("1", "true", "yes"):
            print(f"[npls] {msg}", file=sys.stderr, flush=True)
        if self.debug:
            print(f"info string debug {msg}", flush=True)

    def usi(self) -> None:
        print("id name " + self.name)
        print("option name USI_Ponder type check default false")
        print("option name modelfile type string default " + self.DEFAULT_MODELFILE)
        print("option name gpu_id type spin default " + str(DEFAULT_GPU_ID) + " min -1 max 7")
        print("option name batchsize type spin default " + str(DEFAULT_BATCH_SIZE) + " min 1 max 256")
        print(
            "option name resign_threshold type spin default "
            + str(int(DEFAULT_RESIGN_THRESHOLD * 100))
            + " min 0 max 100"
        )
        print("option name temperature type spin default " + str(int(DEFAULT_TEMPERATURE * 100)) + " min 10 max 1000")
        print("option name time_margin type spin default " + str(DEFAULT_TIME_MARGIN) + " min 0 max 1000")
        print("option name byoyomi_margin type spin default " + str(DEFAULT_BYOYOMI_MARGIN) + " min 0 max 1000")
        print("option name pv_interval type spin default " + str(DEFAULT_PV_INTERVAL) + " min 0 max 10000")
        print(
            "option name activation_function type combo default relu"
            + " var relu var leaky_relu var soft_sin var scaled_arc_tanh var quadratic var cube_root var squash_two"
        )
        print("option name eval_type type combo default resnet var resnet var nnue")
        print("option name debug type check default false")

    def setoption(self, args: list[str]) -> None:
        if args[1] == "modelfile":
            self.modelfile = args[3]
        elif args[1] == "gpu_id":
            self.gpu_id = int(args[3])
        elif args[1] == "batchsize":
            self.batch_size = int(args[3])
        elif args[1] == "resign_threshold":
            self.resign_threshold = int(args[3]) / 100
        elif args[1] == "temperature":
            self.temperature = int(args[3]) / 100
        elif args[1] == "time_margin":
            self.time_margin = int(args[3])
        elif args[1] == "byoyomi_margin":
            self.byoyomi_margin = int(args[3])
        elif args[1] == "pv_interval":
            self.pv_interval = int(args[3])
        elif args[1] == "eval_type":
            self.eval_type = args[3]
        elif args[1] == "activation_function":
            self.activation_function = args[3]
        elif args[1] == "debug":
            self.debug = args[3] == "true"

    def load_evaluator(self) -> None:
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
            f"isready: eval_type={self.eval_type} modelfile={self.modelfile} gpu_id={self.gpu_id} batchsize={self.batch_size}"
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
        self.tree.clear()
        self._log_debug(f"isready: ok device={self.device}")

    def position(self, sfen: str, usi_moves: list[str]) -> None:
        if sfen == "startpos":
            self.root_board.reset()
        elif sfen[:5] == "sfen ":
            self.root_board.set_sfen(sfen[5:])

        for usi_move in usi_moves:
            self.root_board.push_usi(usi_move)

        self.tree.clear()
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
        elif nodes:
            self.halt = nodes
        else:
            self.remaining_time, inc = (btime, binc) if self.root_board.turn == BLACK else (wtime, winc)
            if self.remaining_time is None and byoyomi is None and inc is None:
                self.halt = DEFAULT_CONST_PLAYOUT
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

    def go(self) -> tuple[str, Optional[str]]:
        self.begin_time = time.time()

        if self.root_board.is_game_over():
            return "resign", None
        if self.root_board.is_nyugyoku():
            return "win", None

        if not self.root_board.is_check():
            matemove = self.root_board.mate_move_in_1ply()
            if matemove:
                print("info score mate 1 pv {}".format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None

        matemove = self.root_board.mate_move(3)
        if matemove != 0:
            print("info score mate 3 pv {}".format(move_to_usi(matemove)), flush=True)
            return move_to_usi(matemove), None

        self.playout_count = 0

        legal = list(self.root_board.legal_moves)
        if len(legal) == 1:
            self._log_debug("go: single legal move, skip search")
            return move_to_usi(legal[0]), None

        self._log_debug(f"go: search start legal_moves={len(legal)} halt={self.halt} time_limit_ms={getattr(self, 'time_limit', None)}")
        self.search()
        self._log_debug(f"go: search done playouts={self.playout_count}")

        bestmove_usi, bestvalue, ponder_usi = self.get_bestmove_and_print_pv()
        self._log_debug(f"go: best root_perspective_win={bestvalue:.4f}")

        if bestvalue < self.resign_threshold:
            return "resign", None

        return bestmove_usi, ponder_usi

    def stop(self) -> None:
        self.halt = 0

    def ponderhit(self, last_limits: dict) -> None:
        self.begin_time = time.time()
        self.last_pv_print_time = 0
        self.playout_count = 0
        self.set_limits(**last_limits)

    def quit(self) -> None:
        self.stop()

    # --- NPLS 本探索 ---
    def search(self) -> None:
        self.last_pv_print_time = 0.0
        self._best_node = None
        self._best_value = 0.0
        # NN の value は「その局面の手番側の勝率」。累積は探索開始局面の手番（ルート）から見た勝率に直してから足す。
        self._search_root_turn = self.root_board.turn
        self.tree.recycle_nodes(self.root_board.move_number, 0.0, 1.0)

        if len(self.tree) == 0:
            for move in self.root_board.legal_moves:
                node = NPLSNode(
                    depth=1,
                    moves=[move],
                    value=0.0,
                    total_value=0.0,
                    policy=1.0,
                    total_policy=1.0,
                    board=self.root_board.copy(),
                    priority=0.0,
                )
                node.board.push(move)
                node.priority = node.compute_priority()
                self.tree.push(node)

        while True:
            self.nodes = self.tree.pop_max(self.batch_size)
            if len(self.nodes) == 0:
                break

            if self.evaluator is None:
                raise ValueError("evaluator is None")
            boards = [node.board for node in self.nodes]
            legal_batch = [list(node.board.legal_moves) for node in self.nodes]
            colors = [node.board.turn for node in self.nodes]
            policy_batch, value_logits = self.evaluator.evaluate_batch(
                boards, legal_batch, colors, self.temperature
            )

            for i, (policy_prob, value) in enumerate(zip(policy_batch, value_logits)):
                current_node = self.nodes[i]
                val_side = float(np.asarray(value).reshape(-1)[0])
                val_root = value_from_root_perspective(
                    current_node.board, self._search_root_turn, val_side
                )

                # 合法手のみtreeに追加する
                for j, move in enumerate(current_node.board.legal_moves):
                    nb = current_node.board.copy()
                    nb.push(move)
                    pol_f = float(policy_prob[j]) if j < len(policy_prob) else 0.0
                    new_node = NPLSNode(
                        depth=current_node.depth + 1,
                        moves=current_node.moves + [move],
                        value=val_root,
                        total_value=float(current_node.total_value) + val_root,
                        policy=pol_f,
                        total_policy=float(current_node.total_policy) * pol_f,
                        board=nb,
                        priority=0.0,
                    )
                    # UW　囲碁のカタゴから（valueの最大値最小値の開きからプライオティをあげる）
                    new_node.priority = new_node.compute_priority()
                    self.tree.push(new_node)
                    
                    if self._best_node is None or (new_node.depth % 2 == 0 and self._best_value < float(new_node.total_value / new_node.depth)):
                        self._best_value = float(new_node.total_value)
                        self._best_node = new_node
            
            self.playout_count += len(self.nodes)
            
            if self.check_interruption():
                return

            if self.pv_interval > 0:
                elapsed_ms = int((time.time() - self.begin_time) * 1000)
                if elapsed_ms > self.last_pv_print_time + self.pv_interval:
                    self.last_pv_print_time = elapsed_ms
                    self.get_bestmove_and_print_pv()

    def make_move_label(self, move: int, color: int) -> int:
        return make_move_label(move, color)

    def get_bestmove_and_print_pv(self) -> tuple[str, float, Optional[str]]:
        finish_time = time.time() - self.begin_time

        ponder_usi: Optional[str] = None
        if self._best_node is None:
            lm = list(self.root_board.legal_moves)
            if not lm:
                return "resign", 0.0, None
            bestmove_usi = move_to_usi(lm[0])
            bestvalue = 0.0
            pv = bestmove_usi
        else:
            bestmove_usi = move_to_usi(self._best_node.moves[0])
            raw_total = float(np.asarray(self._best_value).reshape(-1)[0])
            bestvalue = total_value_to_mean_win_prob(raw_total, self._best_node.depth)
            if len(self._best_node.moves) >= 2:
                ponder_usi = move_to_usi(self._best_node.moves[1])
            if self._best_node.moves:
                pv = " ".join(move_to_usi(m) for m in self._best_node.moves)
            else:
                pv = bestmove_usi

        if bestvalue >= 1.0:
            cp = 30000
        elif bestvalue <= 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / max(min(bestvalue, 1.0 - 1e-9), 1e-9) - 1.0) * 600)

        print(
            "info nps {} time {} nodes {} score cp {} pv {}".format(
                int(self.playout_count / finish_time) if finish_time > 0 else 0,
                int(finish_time * 1000),
                self.playout_count,
                cp,
                pv,
            ),
            flush=True,
        )

        sys.stdout.flush()
        return bestmove_usi, bestvalue, ponder_usi

    def check_interruption(self) -> bool:
        if self.halt is not None:
            return self.playout_count >= self.halt

        if self.time_limit is None:
            return False

        spend_ms = int((time.time() - self.begin_time) * 1000)

        if spend_ms * 10 < self.time_limit:
            return False
        if spend_ms < self.minimum_time:
            return False

        return spend_ms >= self.time_limit


if __name__ == "__main__":
    player = NPLSPlayer()
    player.run()
