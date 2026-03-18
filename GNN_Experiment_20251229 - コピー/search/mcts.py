"""
search/mcts.py
モンテカルロ木探索（MCTS）の実装。
shogiAI の uct_node.py と mcts_player.py を統合して移植。
モデルは HybridAlphaZeroNet（model.py）を使用。
"""

import numpy as np
import torch
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

# game/board.py からインポート
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from game.board import (
    FEATURES_SETTINGS,
    FEATURES_NUM,
    MOVE_LABELS_NUM,
    make_move_label,
)
from model.model import create_model

# ===== 定数 =====
DEFAULT_GPU_ID           = 0
DEFAULT_BATCH_SIZE       = 32
DEFAULT_RESIGN_THRESHOLD = 0.01
DEFAULT_C_PUCT           = 1.0
DEFAULT_TEMPERATURE      = 1.0
DEFAULT_TIME_MARGIN      = 1000
DEFAULT_BYOYOMI_MARGIN   = 100
DEFAULT_PV_INTERVAL      = 500
DEFAULT_CONST_PLAYOUT    = 1000
DEFAULT_BLOCKS_CONFIG    = "30blocks"   # create_model() の mode 引数

VALUE_WIN   =  10000
VALUE_LOSE  = -10000
VALUE_DRAW  =  20000
QUEUING     = -1
DISCARDED   = -2
VIRTUAL_LOSS = 1


# ===== ヘルパー =====

def softmax_temperature_with_normalize(
    logits: np.ndarray, temperature: float
) -> "np.ndarray[np.float32, np.float32]":
    """温度パラメータ付きソフトマックスで確率を計算して正規化する。"""
    logits = logits / temperature
    max_logit = logits.max()
    probabilities = np.exp(logits - max_logit)
    probabilities /= probabilities.sum()
    return probabilities


def update_result(current_node: "UctNode", next_index: int, result: float) -> None:
    current_node.sum_value += result
    current_node.move_count += 1 - VIRTUAL_LOSS
    current_node.child_sum_value[next_index] += result
    current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS


# ===== UctNode / NodeTree =====

class UctNode:
    def __init__(self) -> None:
        self.move_count: int = 0
        self.sum_value: float = 0.0
        self.child_move: Optional[list] = None
        self.child_move_count: Optional[np.ndarray] = None
        self.child_sum_value: Optional[np.ndarray] = None
        self.child_node: Optional[list] = None
        self.policy: Optional[np.ndarray] = None
        self.value: Optional[float] = None

    def create_child_node(self, index: int) -> "UctNode":
        self.child_node[index] = UctNode()
        return self.child_node[index]

    def expand_node(self, board: Board) -> None:
        self.child_move = list(board.legal_moves)
        child_num = len(self.child_move)
        self.child_move_count = np.zeros(child_num, dtype=np.int32)
        self.child_sum_value  = np.zeros(child_num, dtype=np.float32)

    def release_children_except_one(self, move: int) -> "UctNode":
        if self.child_move and self.child_node:
            for i in range(len(self.child_move)):
                if self.child_move[i] == move:
                    if self.child_node[i] is None:
                        self.child_node[i] = UctNode()
                    if len(self.child_move) > 1:
                        self.child_move       = [move]
                        self.child_move_count = None
                        self.child_sum_value  = None
                        self.policy           = None
                        self.child_node       = [self.child_node[i]]
                    return self.child_node[0]

        self.child_move       = [move]
        self.child_move_count = None
        self.child_sum_value  = None
        self.policy           = None
        self.child_node       = [UctNode()]
        return self.child_node[0]


class NodeTree:
    def __init__(self) -> None:
        self.current_head: Optional[UctNode] = None
        self.gamebegin_node: Optional[UctNode] = None
        self.history_starting_pos_key: Optional[int] = None

    def reset_to_position(self, starting_pos_key: int, moves: list) -> None:
        if self.history_starting_pos_key != starting_pos_key:
            self.gamebegin_node = UctNode()
            self.current_head   = self.gamebegin_node

        self.history_starting_pos_key = starting_pos_key

        old_head  = self.current_head
        prev_head = None
        self.current_head = self.gamebegin_node
        seen_old_head = (self.gamebegin_node == old_head)

        for move in moves:
            prev_head = self.current_head
            self.current_head = self.current_head.release_children_except_one(move)
            if old_head == self.current_head:
                seen_old_head = True

        if not seen_old_head and self.current_head != old_head:
            if prev_head:
                assert len(prev_head.child_move) == 1
                prev_head.child_node[0] = UctNode()
                self.current_head = prev_head.child_node[0]
            else:
                self.gamebegin_node = UctNode()
                self.current_head   = self.gamebegin_node


# ===== 評価待ちキュー要素 =====

class EvalQueueElement:
    def __init__(self) -> None:
        self.node:  Optional[UctNode] = None
        self.color: Optional[int]     = None

    def set(self, node: UctNode, color: int) -> None:
        self.node  = node
        self.color = color


# ===== MCTSPlayer =====

class MCTSPlayer:
    """
    USI対局エンジン。
    HybridAlphaZeroNet（model.py）を評価関数に使ったMCTSプレイヤー。
    """

    name = "GNN-Shogi"
    DEFAULT_MODELFILE = "weights/checkpoint.pth"

    def __init__(
        self,
        features_mode:            int  = 0,
        blocks_config_mode:       str  = DEFAULT_BLOCKS_CONFIG,
        name:                     str  = "GNN-Shogi",
        modelfile:                str  = "weights/checkpoint.pth",
    ) -> None:
        self.name:                 str  = name
        self.modelfile:            str  = modelfile
        self.model                      = None
        self.features:             Optional[torch.Tensor] = None
        self.eval_queue:           Optional[list]         = None
        self.current_batch_index:  int  = 0

        self.root_board: Board    = Board()
        self.tree: NodeTree       = NodeTree()

        self.playout_count: int   = 0
        self.halt:          Optional[int] = None

        self.gpu_id:    int                              = DEFAULT_GPU_ID
        self.device:    Optional[Union[str, torch.device, int]] = None
        self.batch_size: int                             = DEFAULT_BATCH_SIZE

        self.resign_threshold: float = DEFAULT_RESIGN_THRESHOLD
        self.c_puct:           float = DEFAULT_C_PUCT
        self.temperature:      float = DEFAULT_TEMPERATURE
        self.time_margin:      int   = DEFAULT_TIME_MARGIN
        self.byoyomi_margin:   int   = DEFAULT_BYOYOMI_MARGIN
        self.pv_interval:      int   = DEFAULT_PV_INTERVAL

        self.features_setting  = FEATURES_SETTINGS[features_mode]
        self.blocks_config_mode = blocks_config_mode
        self.debug = False

    # ---- USIコマンド対応 ----

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
        print("option name c_puct type spin default " + str(int(DEFAULT_C_PUCT * 100)) + " min 10 max 1000")
        print(
            "option name temperature type spin default "
            + str(int(DEFAULT_TEMPERATURE * 100))
            + " min 10 max 1000"
        )
        print("option name time_margin type spin default " + str(DEFAULT_TIME_MARGIN) + " min 0 max 1000")
        print("option name byoyomi_margin type spin default " + str(DEFAULT_BYOYOMI_MARGIN) + " min 0 max 1000")
        print("option name pv_interval type spin default " + str(DEFAULT_PV_INTERVAL) + " min 0 max 10000")
        print("option name debug type check default false")

    def setoption(self, args: list) -> None:
        if args[1] == "modelfile":
            self.modelfile = args[3]
        elif args[1] == "gpu_id":
            self.gpu_id = int(args[3])
        elif args[1] == "batchsize":
            self.batch_size = int(args[3])
        elif args[1] == "resign_threshold":
            self.resign_threshold = int(args[3]) / 100
        elif args[1] == "c_puct":
            self.c_puct = int(args[3]) / 100
        elif args[1] == "temperature":
            self.temperature = int(args[3]) / 100
        elif args[1] == "time_margin":
            self.time_margin = int(args[3])
        elif args[1] == "byoyomi_margin":
            self.byoyomi_margin = int(args[3])
        elif args[1] == "pv_interval":
            self.pv_interval = int(args[3])
        elif args[1] == "debug":
            self.debug = args[3] == "true"

    def load_model(self) -> None:
        """HybridAlphaZeroNet をロードする。"""
        self.model = create_model(
            input_channels=self.features_setting.features_num,
            num_actions=MOVE_LABELS_NUM,
            mode=self.blocks_config_mode,
        )
        self.model.to(self.device)
        if os.path.exists(self.modelfile):
            checkpoint = torch.load(self.modelfile, map_location=self.device)
            # チェックポイント形式によって読み込みキーを切り替える
            state_dict = checkpoint.get("model", checkpoint)
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def init_features(self) -> None:
        self.features = torch.empty(
            (self.batch_size, self.features_setting.features_num, 9, 9),
            dtype=torch.float32,
            pin_memory=(self.gpu_id >= 0),
        )

    def isready(self) -> None:
        if self.gpu_id >= 0:
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.load_model()
        self.root_board.reset()
        self.tree.reset_to_position(self.root_board.zobrist_hash(), [])

        self.init_features()
        self.eval_queue = [EvalQueueElement() for _ in range(self.batch_size)]
        self.current_batch_index = 0

        # 初回推論でモデルをキャッシュ
        current_node = self.tree.current_head
        current_node.expand_node(self.root_board)
        for _ in range(self.batch_size):
            self.queue_node(self.root_board, current_node)
        self.eval_node()

    def position(self, sfen: str, usi_moves: list) -> None:
        if sfen == "startpos":
            self.root_board.reset()
        elif sfen[:5] == "sfen ":
            self.root_board.set_sfen(sfen[5:])

        starting_pos_key = self.root_board.zobrist_hash()
        moves = []
        for usi_move in usi_moves:
            move = self.root_board.push_usi(usi_move)
            moves.append(move)
        self.tree.reset_to_position(starting_pos_key, moves)

        if self.debug:
            print(self.root_board)

    def set_limits(
        self,
        btime:    Optional[int]  = None,
        wtime:    Optional[int]  = None,
        byoyomi:  Optional[int]  = None,
        binc:     Optional[int]  = None,
        winc:     Optional[int]  = None,
        nodes:    Optional[int]  = None,
        infinite: bool           = False,
        ponder:   bool           = False,
    ) -> None:
        if infinite or ponder:
            self.halt = 2**31 - 1
        elif nodes:
            self.halt = nodes
        else:
            remaining_time, inc = (btime, binc) if self.root_board.turn == BLACK else (wtime, winc)
            if remaining_time is None and byoyomi is None and inc is None:
                self.halt = DEFAULT_CONST_PLAYOUT
            else:
                self.minimum_time    = 0
                remaining_time       = int(remaining_time) if remaining_time else 0
                inc                  = int(inc) if inc else 0
                self.time_limit      = remaining_time / (14 + max(0, 30 - self.root_board.move_number)) + inc
                self.remaining_time  = remaining_time
                if byoyomi:
                    byoyomi          = int(byoyomi) - self.byoyomi_margin
                    self.minimum_time = byoyomi
                    if self.time_limit < byoyomi:
                        self.time_limit = byoyomi
                self.extend_time = self.time_limit > self.minimum_time
                self.halt = None

    def go(self) -> tuple:
        self.begin_time = time.time()

        if self.root_board.is_game_over():
            return "resign", None
        if self.root_board.is_nyugyoku():
            return "win", None

        current_node = self.tree.current_head

        if current_node.value == VALUE_WIN:
            matemove = self.root_board.mate_move(3)
            if matemove != 0:
                print("info score mate 3 pv {}".format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None
        if not self.root_board.is_check():
            matemove = self.root_board.mate_move_in_1ply()
            if matemove:
                print("info score mate 1 pv {}".format(move_to_usi(matemove)), flush=True)
                return move_to_usi(matemove), None

        self.playout_count = 0

        if current_node.child_move is None:
            current_node.expand_node(self.root_board)

        if self.halt is None and len(current_node.child_move) == 1:
            if current_node.child_move_count[0] > 0:
                bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()
                return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None
            else:
                return move_to_usi(current_node.child_move[0]), None

        if current_node.policy is None:
            self.current_batch_index = 0
            self.queue_node(self.root_board, current_node)
            self.eval_node()

        self.search()

        bestmove, bestvalue, ponder_move = self.get_bestmove_and_print_pv()

        if self.debug:
            for i in range(len(current_node.child_move)):
                print(
                    "{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}".format(
                        i,
                        move_to_usi(current_node.child_move[i]),
                        current_node.child_move_count[i],
                        current_node.policy[i],
                        current_node.child_sum_value[i] / current_node.child_move_count[i]
                        if current_node.child_move_count[i] > 0 else 0,
                    )
                )

        if bestvalue < self.resign_threshold:
            return "resign", None

        return move_to_usi(bestmove), move_to_usi(ponder_move) if ponder_move else None

    def stop(self) -> None:
        self.halt = 0

    def ponderhit(self, last_limits: dict) -> None:
        self.begin_time       = time.time()
        self.last_pv_print_time = 0
        self.playout_count    = 0
        self.set_limits(**last_limits)

    def quit(self) -> None:
        self.stop()

    # ---- 探索 ----

    def search(self) -> None:
        self.last_pv_print_time: float = 0
        trajectories_batch           = []
        trajectories_batch_discarded = []

        while True:
            trajectories_batch.clear()
            trajectories_batch_discarded.clear()
            self.current_batch_index = 0

            for i in range(self.batch_size):
                board = self.root_board.copy()
                trajectories_batch.append([])
                result = self.uct_search(board, self.tree.current_head, trajectories_batch[-1])

                if result != DISCARDED:
                    self.playout_count += 1
                else:
                    trajectories_batch_discarded.append(trajectories_batch[-1])
                    if len(trajectories_batch_discarded) > self.batch_size // 2:
                        trajectories_batch.pop()
                        break

                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()

            if len(trajectories_batch) > 0:
                self.eval_node()

            for trajectories in trajectories_batch_discarded:
                for current_node, next_index in trajectories:
                    current_node.move_count -= VIRTUAL_LOSS
                    current_node.child_move_count[next_index] -= VIRTUAL_LOSS

            for trajectories in trajectories_batch:
                result = None
                for current_node, next_index in reversed(trajectories):
                    if result is None:
                        result = 1.0 - current_node.child_node[next_index].value
                    update_result(current_node, next_index, result)
                    result = 1.0 - result

            if self.check_interruption():
                return

            if self.pv_interval > 0:
                elapsed_time = int((time.time() - self.begin_time) * 1000)
                if elapsed_time > self.last_pv_print_time + self.pv_interval:
                    self.last_pv_print_time = elapsed_time
                    self.get_bestmove_and_print_pv()

    def uct_search(self, board: Board, current_node: UctNode, trajectories: list) -> float:
        if not current_node.child_node:
            current_node.child_node = [None] * len(current_node.child_move)

        next_index = self.select_max_ucb_child(current_node)
        board.push(current_node.child_move[next_index])

        current_node.move_count += VIRTUAL_LOSS
        current_node.child_move_count[next_index] += VIRTUAL_LOSS
        trajectories.append((current_node, next_index))

        if current_node.child_node[next_index] is None:
            child_node = current_node.create_child_node(next_index)

            draw = board.is_draw()
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    child_node.value = VALUE_DRAW
                    result = 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    child_node.value = VALUE_LOSE
                    result = 1.0
            else:
                if board.is_nyugyoku() or board.mate_move(3):
                    child_node.value = VALUE_WIN
                    result = 0.0
                else:
                    child_node.expand_node(board)
                    if len(child_node.child_move) == 0:
                        child_node.value = VALUE_LOSE
                        result = 1.0
                    else:
                        self.queue_node(board, child_node)
                        return QUEUING
        else:
            next_node = current_node.child_node[next_index]
            if next_node.value is None:
                return DISCARDED
            if next_node.value == VALUE_WIN:
                result = 0.0
            elif next_node.value == VALUE_LOSE:
                result = 1.0
            elif next_node.value == VALUE_DRAW:
                result = 0.5
            elif len(next_node.child_move) == 0:
                result = 1.0
            else:
                result = self.uct_search(board, next_node, trajectories)

        if result in (QUEUING, DISCARDED):
            return result

        update_result(current_node, next_index, result)
        return 1.0 - result

    def select_max_ucb_child(self, node: UctNode):
        q = np.divide(
            node.child_sum_value,
            node.child_move_count,
            out=np.zeros(len(node.child_move), np.float32),
            where=node.child_move_count != 0,
        )
        if node.move_count == 0:
            u = 1.0
        else:
            u = np.sqrt(np.float32(node.move_count)) / (1 + node.child_move_count)
        ucb = q + self.c_puct * u * node.policy
        return int(np.argmax(ucb))

    def get_bestmove_and_print_pv(self) -> tuple:
        finish_time  = time.time() - self.begin_time
        current_node = self.tree.current_head
        selected_index = int(np.argmax(current_node.child_move_count))

        bestvalue = (
            current_node.child_sum_value[selected_index]
            / current_node.child_move_count[selected_index]
        )
        bestmove = current_node.child_move[selected_index]

        if bestvalue == 1.0:
            cp = 30000
        elif bestvalue == 0.0:
            cp = -30000
        else:
            cp = int(-math.log(1.0 / bestvalue - 1.0) * 600)

        pv = move_to_usi(bestmove)
        ponder_move = None
        pv_node = current_node
        while pv_node.child_node:
            pv_node = pv_node.child_node[selected_index]
            if pv_node is None or pv_node.child_move is None or pv_node.move_count == 0:
                break
            selected_index = int(np.argmax(pv_node.child_move_count))
            pv += " " + move_to_usi(pv_node.child_move[selected_index])
            if ponder_move is None:
                ponder_move = pv_node.child_move[selected_index]

        print(
            "info nps {} time {} nodes {} score cp {} pv {}".format(
                int(self.playout_count / finish_time) if finish_time > 0 else 0,
                int(finish_time * 1000),
                current_node.move_count,
                cp,
                pv,
            ),
            flush=True,
        )
        return bestmove, bestvalue, ponder_move

    def check_interruption(self) -> bool:
        if self.halt is not None:
            return self.playout_count >= self.halt

        current_node = self.tree.current_head
        if (
            current_node is not None
            and current_node.child_move is not None
            and len(current_node.child_move) == 1
        ):
            return True

        spend_time = int((time.time() - self.begin_time) * 1000)
        if spend_time * 10 < self.time_limit or spend_time < self.minimum_time:
            return False

        child_move_count = current_node.child_move_count
        second_index, first_index = np.argpartition(child_move_count, -2)[-2:]
        second, first = child_move_count[[second_index, first_index]]
        rest = int(self.playout_count * ((self.time_limit - spend_time) / spend_time))

        if first - second <= rest:
            return False

        if (
            self.extend_time
            and self.root_board.move_number > 20
            and self.remaining_time > self.time_limit * 2
            and (
                first < second * 1.5
                or current_node.child_sum_value[first_index] / child_move_count[first_index]
                < current_node.child_sum_value[second_index] / child_move_count[second_index]
            )
        ):
            self.time_limit *= 2
            self.extend_time = False
            print("info string extend_time")
            return False

        return True

    # ---- 推論 ----

    def make_input_features(self, board: Board) -> None:
        features_numpy = self.features.numpy()
        self.features_setting.make_features(board, features_numpy[self.current_batch_index])

    def queue_node(self, board: Board, node: UctNode) -> None:
        self.make_input_features(board)
        if self.eval_queue is None:
            raise ValueError("eval_queue is None")
        self.eval_queue[self.current_batch_index].set(node, board.turn)
        self.current_batch_index += 1

    def infer(self) -> tuple:
        """
        HybridAlphaZeroNet で推論する。
        policy: log_softmax → exp() で確率に変換
        value:  Tanh [-1,1] → (v+1)/2 で [0,1] に正規化
        """
        with torch.no_grad():
            if self.features is None or self.model is None:
                raise ValueError("features or model is None")
            x = self.features[0: self.current_batch_index].to(self.device)
            log_policy, value = self.model(x)
            # log_softmax → softmax 確率
            policy = torch.exp(log_policy).cpu().numpy()
            # Tanh [-1,1] → [0,1]
            win_rate = ((value + 1.0) / 2.0).cpu().numpy()
            return policy, win_rate

    def eval_node(self) -> None:
        policy_batch, values = self.infer()

        for i, (policy, value) in enumerate(zip(policy_batch, values)):
            current_node = self.eval_queue[i].node
            color        = self.eval_queue[i].color

            legal_move_probabilities = np.empty(len(current_node.child_move), dtype=np.float32)
            for j, move in enumerate(current_node.child_move):
                move_label = make_move_label(move, color)
                legal_move_probabilities[j] = policy[move_label]

            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, self.temperature)
            current_node.policy = probabilities
            current_node.value  = float(value)
