from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cshogi
from cshogi import Board, BLACK, WHITE, PIECE_TYPES


@dataclass(frozen=True)
class HalfKAv2Spec:
    board_squares: int = 81
    # cshogi の駒種数（盤上は piece_to_piece_type で 0..len-1 に正規化する）
    piece_types: int = len(PIECE_TYPES)
    hand_buckets: int = 8

    @property
    def board_feature_dim(self) -> int:
        return self.board_squares * self.piece_types * 2

    @property
    def hand_feature_dim(self) -> int:
        return 2 * 7 * self.hand_buckets

    @property
    def total_feature_dim(self) -> int:
        return self.board_feature_dim + self.hand_feature_dim


SPEC = HalfKAv2Spec()


def _rotate_sq_if_white(square: int, color: int) -> int:
    return 80 - square if color == WHITE else square


def _piece_type_index(piece: int) -> int:
    """駒種 0..len(PIECE_TYPES)-1。

    cshogi の盤上駒IDは成りで 17..24 などになり得るため ``abs(piece)-1`` は不可。
    ``piece_to_piece_type`` は空で 0、駒種は通常 1..14 を返す。
    """
    t = cshogi.piece_to_piece_type(piece)
    if t <= 0:
        return 0
    if t <= len(PIECE_TYPES):
        return t - 1
    return len(PIECE_TYPES) - 1


def _color_index(piece: int, perspective_color: int) -> int:
    piece_color = BLACK if piece > 0 else WHITE
    return 0 if piece_color == perspective_color else 1


def extract_halfkav2_indices(board: Board, perspective_color: int) -> np.ndarray:
    """HalfKAv2 互換の簡易疎特徴（フル再計算版）。

    参考実装にある差分更新はまだ行わず、各局面でインデックス列を再構築する。
    """
    board_array = board.pieces
    out: list[int] = []

    for sq in range(81):
        piece = int(board_array[sq])
        if piece == 0:
            continue
        sq_p = _rotate_sq_if_white(sq, perspective_color)
        pt = _piece_type_index(piece)
        co = _color_index(piece, perspective_color)
        idx = (co * SPEC.piece_types + pt) * SPEC.board_squares + sq_p
        out.append(idx)

    base = SPEC.board_feature_dim
    hands = board.pieces_in_hand
    own_hands = hands[perspective_color]
    opp_hands = hands[1 - perspective_color]

    # hand order: pawn,lance,knight,silver,gold,bishop,rook
    for i, n in enumerate(own_hands):
        for k in range(min(int(n), SPEC.hand_buckets)):
            out.append(base + i * SPEC.hand_buckets + k)
    base += 7 * SPEC.hand_buckets
    for i, n in enumerate(opp_hands):
        for k in range(min(int(n), SPEC.hand_buckets)):
            out.append(base + i * SPEC.hand_buckets + k)

    if not out:
        # EmbeddingBag の空入力回避
        out.append(0)
    arr = np.asarray(out, dtype=np.int64)
    return np.clip(arr, 0, SPEC.total_feature_dim - 1)
