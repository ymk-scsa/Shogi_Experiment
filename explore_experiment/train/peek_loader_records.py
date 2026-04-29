import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math
import numpy as np
import typer
from cshogi import Board, HuffmanCodedPosAndEval, PackedSfenValue, move16_from_psv, move_to_usi
from typing_extensions import Annotated

from shogi.feature import make_move_label, make_result

app = typer.Typer()


def _psv_score_to_value(score: float, turn: int, scale: float) -> float:
    v = 1.0 / (1.0 + math.exp(-float(score) / scale))
    if turn != 0:
        v = 1.0 - v
    return float(v)


@app.command()
def run(
    data_path: Annotated[str, typer.Option(help="Path to HCPE or PSV file")],
    data_format: Annotated[Literal["hcpe", "psv"], typer.Option(help="hcpe or psv")] = "hcpe",
    start: Annotated[int, typer.Option(help="Start index")] = 0,
    count: Annotated[int, typer.Option(help="How many records to print")] = 10,
    psv_score_scale: Annotated[float, typer.Option(help="PSV score sigmoid scale")] = 600.0,
    summary_only: Annotated[bool, typer.Option(help="Print only summary counts")] = False,
) -> None:
    board = Board()
    fmt = data_format.lower()
    if fmt == "hcpe":
        data = np.fromfile(data_path, dtype=HuffmanCodedPosAndEval)
    else:
        data = np.fromfile(data_path, dtype=PackedSfenValue)

    end = min(len(data), start + count)
    print(f"format={fmt} total={len(data)} range=[{start}, {end})")

    if fmt == "hcpe":
        sliced = data[start:end]
        game_results = sliced["gameResult"].astype(np.int64)
        uniq, cnt = np.unique(game_results, return_counts=True)
        print("summary(gameResult counts):")
        for k, v in zip(uniq.tolist(), cnt.tolist()):
            print(f"  gameResult={k}: {v}")
        print(f"  gameResult=0 ratio: {float(np.mean(game_results == 0)):.6f}")
    else:
        sliced = data[start:end]
        scores = sliced["score"].astype(np.float64)
        values = []
        for rec in sliced:
            board.set_psfen(rec["sfen"])
            values.append(_psv_score_to_value(float(rec["score"]), board.turn, psv_score_scale))
        v_arr = np.asarray(values, dtype=np.float64)
        print("summary(score/value stats):")
        print(
            f"  score min={float(scores.min()):.2f} max={float(scores.max()):.2f} "
            f"mean={float(scores.mean()):.2f} std={float(scores.std()):.2f}"
        )
        print(
            f"  value min={float(v_arr.min()):.6f} max={float(v_arr.max()):.6f} "
            f"mean={float(v_arr.mean()):.6f} std={float(v_arr.std()):.6f}"
        )

    if summary_only:
        return

    for idx in range(start, end):
        rec = data[idx]
        if fmt == "hcpe":
            board.set_hcp(rec["hcp"])
            move16 = int(rec["bestMove16"])
            value = float(make_result(int(rec["gameResult"]), board.turn))
            raw_value = int(rec["gameResult"])
            extra = f"gameResult(raw)={raw_value}"
        else:
            board.set_psfen(rec["sfen"])
            move16 = int(move16_from_psv(int(rec["move"])))
            value = _psv_score_to_value(float(rec["score"]), board.turn, psv_score_scale)
            raw_score = int(rec["score"])
            extra = f"score(raw)={raw_score}"

        move_usi = move_to_usi(move16)
        legal_moves = set(board.legal_moves)
        # move16(16bit教師手) と board.legal_moves の内部表現は別なので、USI経由で比較する
        move_internal_from_usi = board.move_from_usi(move_usi)
        is_legal_by_internal = move_internal_from_usi in legal_moves

        teacher_label = make_move_label(move16, board.turn)
        legal_labels = {make_move_label(m, board.turn) for m in legal_moves}
        is_legal_by_label = teacher_label in legal_labels

        print("-" * 80)
        print(
            f"index={idx} turn={'BLACK' if board.turn == 0 else 'WHITE'} "
            f"legal_by_internal={is_legal_by_internal} legal_by_label={is_legal_by_label}"
        )
        print(f"sfen={board.sfen()}")
        print(f"teacher_move16={move16} teacher_usi={move_usi} teacher_label={teacher_label}")
        print(f"value={value:.6f} {extra}")


if __name__ == "__main__":
    app()
