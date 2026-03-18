"""
game/board.py
将棋の盤面操作・入力特徴量生成・指し手ラベル変換を担当するモジュール。
shogiAI (Shogi202050504/shogiAI) の features.py および moves.py を
GNN_Experiment_20251229 用にまとめて移植。
"""

from typing import Callable
import cshogi
from cshogi import (
    BLACK,
    WHITE,
    PIECE_TYPES,
    MAX_PIECES_IN_HAND,
    HAND_PIECES,
    move_is_drop,
    move_to,
    move_from,
    move_is_promotion,
    move_drop_hand_piece,
    BLACK_WIN,
    WHITE_WIN,
)
import numpy as np

# ===== 移動方向定数 =====
MOVE_DIRECTION = [
    UP,
    UP_LEFT,
    UP_RIGHT,
    LEFT,
    RIGHT,
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    UP2_LEFT,
    UP2_RIGHT,
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE,
] = range(20)

# 座標オフセット（利き/ヒモ計算用）
MOVE_OFFSET = {
    UP: (-1, 0),
    UP_LEFT: (-1, -1),
    UP_RIGHT: (-1, 1),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    DOWN_LEFT: (1, -1),
    DOWN_RIGHT: (1, 1),
    UP2_LEFT: (-2, -1),
    UP2_RIGHT: (-2, 1),
}

# 駒種ごとの移動方向（1マス移動, 遠移動）
PIECE_DIRECTIONS = {
    cshogi.PAWN:        ([UP], []),
    cshogi.LANCE:       ([], [UP]),
    cshogi.KNIGHT:      ([UP2_LEFT, UP2_RIGHT], []),
    cshogi.SILVER:      ([UP_RIGHT, UP, DOWN_RIGHT, UP_LEFT, DOWN_LEFT], []),
    cshogi.GOLD:        ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.BISHOP:      ([], [UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT]),
    cshogi.ROOK:        ([], [UP, DOWN, RIGHT, LEFT]),
    cshogi.KING:        ([UP_RIGHT, UP, DOWN_RIGHT, RIGHT, LEFT, UP_LEFT, DOWN, DOWN_LEFT], []),
    cshogi.PROM_PAWN:   ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_LANCE:  ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_KNIGHT: ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_SILVER: ([UP_RIGHT, UP, UP_LEFT, RIGHT, LEFT, DOWN], []),
    cshogi.PROM_BISHOP: ([UP, DOWN, RIGHT, LEFT], [UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT]),
    cshogi.PROM_ROOK:   ([UP_RIGHT, DOWN_RIGHT, UP_LEFT, DOWN_LEFT], [UP, DOWN, RIGHT, LEFT]),
}

# ===== 特徴量数・ラベル数 定数 =====
# 盤上の駒プレーン数（28 = 14種 × 2色）+ 持ち駒プレーン数（18）= 46
FEATURES_NUM           = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2    # 46
DIRECTION_NUM          = len(MOVE_OFFSET)                                         # 10

FEATURES_KIKI_NUM      = FEATURES_NUM + DIRECTION_NUM * 2                         # 66
FEATURES_HIMO_NUM      = FEATURES_NUM + DIRECTION_NUM * 2                         # 66
FEATURES_SMALL_NUM     = len(PIECE_TYPES) * 2 + len(MAX_PIECES_IN_HAND) * 2      # 42

# 方策ラベル空間
MOVE_PLANES_NUM  = len(MOVE_DIRECTION) + len(HAND_PIECES)   # 20 + 7 = 27
MOVE_LABELS_NUM  = MOVE_PLANES_NUM * 81                      # 2187

FEATURES_MODE = [
    FEATURES_DEFAULT,
    FEATURES_KIKI,
    FEATURES_HIMO,
    FEATURES_SMALL,
] = range(4)

# 利き/ヒモ特徴の先頭インデックス
_BASE_INDEX = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2


# ===== 入力特徴量生成 =====

def make_input_features(board: cshogi.Board, features: np.ndarray) -> None:
    """標準の入力特徴量（46ch）を生成する。"""
    features.fill(0)
    if board.turn == BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    # 持ち駒
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, MAX_PIECES_IN_HAND):
            features[i: i + num].fill(1)
            i += max_num


def make_kiki_features(board: cshogi.Board, features: np.ndarray) -> None:
    """駒の利きを特徴量に追加する（kikiモード）。"""
    attack_board = board.copy()
    if board.turn != BLACK:
        attack_board = cshogi.Board(sfen=attack_board.sfen())

    for square, piece in enumerate(attack_board.pieces):
        piece_type = cshogi.piece_to_piece_type(piece)
        if piece_type:
            piece_color = 0 if piece == piece_type else 1
            piece_direction_list = PIECE_DIRECTIONS.get(piece_type, ([], []))
            is_own = piece_color == board.turn
            direction_offset = _BASE_INDEX + (0 if is_own else DIRECTION_NUM)

            # 1マス移動の利き
            for direction in piece_direction_list[0]:
                move_y, move_x = MOVE_OFFSET[direction]
                x = square // 9 + move_x
                y = square % 9 + move_y * (-1 if not is_own else 1)
                if 0 <= y < 9 and 0 <= x < 9:
                    features[direction_offset + direction][x][y] = 1

            # 遠移動の利き
            for direction in piece_direction_list[1]:
                move_y, move_x = MOVE_OFFSET[direction]
                x = square // 9
                y = square % 9
                while True:
                    x += move_x
                    y += move_y * (-1 if not is_own else 1)
                    if not (0 <= y < 9 and 0 <= x < 9):
                        break
                    features[direction_offset + direction][x][y] = 1
                    if attack_board.piece_type(y + x * 9) != 0:
                        break


def make_himo_features(board: cshogi.Board, features: np.ndarray) -> None:
    """駒のヒモ（紐付き）を特徴量に追加する（himoモード）。"""
    attack_board = board.copy()
    if board.turn != BLACK:
        attack_board = cshogi.Board(sfen=attack_board.sfen())

    for square, piece in enumerate(attack_board.pieces):
        piece_type = cshogi.piece_to_piece_type(piece)
        if piece_type:
            piece_color = 0 if piece == piece_type else 1
            piece_direction_list = PIECE_DIRECTIONS.get(piece_type, ([], []))
            is_own = piece_color == board.turn
            direction_offset = _BASE_INDEX + (0 if is_own else DIRECTION_NUM)

            # 1マス移動の紐付き
            for direction in piece_direction_list[0]:
                move_y, move_x = MOVE_OFFSET[direction]
                x = square // 9 + move_x
                y = square % 9 + move_y * (-1 if not is_own else 1)
                if 0 <= y < 9 and 0 <= x < 9 and attack_board.piece_type(y + x * 9) != 0:
                    features[direction_offset + direction][x][y] = 1

            # 遠移動の紐付き
            for direction in piece_direction_list[1]:
                move_y, move_x = MOVE_OFFSET[direction]
                x = square // 9
                y = square % 9
                while True:
                    x += move_x
                    y += move_y * (-1 if not is_own else 1)
                    if not (0 <= y < 9 and 0 <= x < 9):
                        break
                    if attack_board.piece_type(y + x * 9) != 0:
                        features[direction_offset + direction][x][y] = 1
                        break


def make_input_features_kiki(board: cshogi.Board, features: np.ndarray) -> None:
    """標準特徴量 + 利き特徴量を生成する（kikiモード, 66ch）。"""
    make_input_features(board, features)
    make_kiki_features(board, features)


def make_input_features_himo(board: cshogi.Board, features: np.ndarray) -> None:
    """標準特徴量 + ヒモ特徴量を生成する（himoモード, 66ch）。"""
    make_input_features(board, features)
    make_himo_features(board, features)


def make_input_features_small(board: cshogi.Board, features: np.ndarray) -> None:
    """小型の入力特徴量（smallモード, 42ch）を生成する。"""
    features.fill(0)
    if board.turn == BLACK:
        board.piece_planes(features)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(features)
        pieces_in_hand = reversed(board.pieces_in_hand)
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, MAX_PIECES_IN_HAND):
            fill = 81 * num // max_num - 1
            features[i][0: fill // 9].fill(1)
            features[i][fill // 9][0: fill % 9].fill(1)
            i += 1


# ===== FeaturesSetting =====

class FeaturesSetting:
    """特徴量生成設定をまとめるクラス。"""
    def __init__(
        self,
        make_features: Callable[[cshogi.Board, np.ndarray], None] = make_input_features,
        features_num: int = FEATURES_NUM,
    ) -> None:
        self.make_features = make_features
        self.features_num = features_num


FEATURES_SETTINGS = {
    FEATURES_DEFAULT: FeaturesSetting(),
    FEATURES_KIKI: FeaturesSetting(
        make_features=make_input_features_kiki,
        features_num=FEATURES_KIKI_NUM,
    ),
    FEATURES_HIMO: FeaturesSetting(
        make_features=make_input_features_himo,
        features_num=FEATURES_HIMO_NUM,
    ),
    FEATURES_SMALL: FeaturesSetting(
        make_features=make_input_features_small,
        features_num=FEATURES_SMALL_NUM,
    ),
}


# ===== 指し手ラベル変換 =====

def make_move_label(move: int, color: int) -> int:
    """
    指し手を方策ネットワークの出力インデックスに変換する。
    返り値の範囲: 0 〜 MOVE_LABELS_NUM-1 (= 2186)
    """
    if not move_is_drop(move):
        to_sq: int = move_to(move)
        from_sq: int = move_from(move)
        if color == WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y

        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_LEFT
            elif dir_x > 0:
                move_direction = UP_RIGHT
            else:
                move_direction = UP_LEFT
        elif dir_y == 0:
            if dir_x > 0:
                move_direction = RIGHT
            else:
                move_direction = LEFT
        else:
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x > 0:
                move_direction = DOWN_RIGHT
            else:
                move_direction = DOWN_LEFT

        if move_is_promotion(move):
            move_direction += 10
    else:
        to_sq = move_to(move)
        if color == WHITE:
            to_sq = 80 - to_sq
        move_direction = len(MOVE_DIRECTION) + move_drop_hand_piece(move)

    return int(move_direction * 81 + to_sq)


# ===== 対局結果 → 報酬変換 =====

def make_result(game_result: int, color: int) -> float:
    """
    対局結果（BLACK_WIN / WHITE_WIN）を手番側の報酬 (0.0〜1.0) に変換する。
    引き分けは 0.5。
    """
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1.0
        if game_result == WHITE_WIN:
            return 0.0
    else:
        if game_result == BLACK_WIN:
            return 0.0
        if game_result == WHITE_WIN:
            return 1.0
    return 0.5
