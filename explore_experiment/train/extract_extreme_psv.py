import sys
from pathlib import Path

# train/ から直接実行したときにプロジェクトルートを import パスに含める
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math
import os

import numpy as np
import typer
from cshogi import Board, PackedSfenValue
from typing_extensions import Annotated

from util.directory import ensure_directory_exists

app = typer.Typer()


def score_to_value(score: float, turn: int, scale: float) -> float:
    """PSV score(先手視点 cp) -> value[0,1] へ変換し、後手番なら反転する。"""
    v = 1.0 / (1.0 + math.exp(-float(score) / scale))
    if turn != 0:
        v = 1.0 - v
    return float(v)


@app.command()
def run(
    input_path: Annotated[str, typer.Option(help="Input PSV file path")],
    output_path: Annotated[str, typer.Option(help="Output PSV file path")],
    low_threshold: Annotated[float, typer.Option(help="Extract if value <= this")] = 0.1,
    high_threshold: Annotated[float, typer.Option(help="Extract if value >= this")] = 0.9,
    score_scale: Annotated[float, typer.Option(help="Sigmoid scale for score->value")] = 600.0,
    chunk_size: Annotated[int, typer.Option(help="Records per chunk")] = 200000,
    overwrite: Annotated[bool, typer.Option(help="Overwrite output file if exists")] = True,
) -> None:
    if not (0.0 <= low_threshold <= 1.0 and 0.0 <= high_threshold <= 1.0):
        raise ValueError("low_threshold/high_threshold must be in [0,1]")
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must be <= high_threshold")
    if score_scale <= 0:
        raise ValueError("score_scale must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    ensure_directory_exists(output_path)
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)
    elif os.path.exists(output_path):
        raise FileExistsError(f"Output already exists: {output_path}")

    data = np.memmap(input_path, dtype=PackedSfenValue, mode="r")
    total = len(data)
    board = Board()

    kept_total = 0
    for begin in range(0, total, chunk_size):
        end = min(begin + chunk_size, total)
        chunk = data[begin:end]
        keep_flags = np.zeros(len(chunk), dtype=np.bool_)

        for i, rec in enumerate(chunk):
            board.set_psfen(rec["sfen"])
            value = score_to_value(float(rec["score"]), board.turn, score_scale)
            keep_flags[i] = value <= low_threshold or value >= high_threshold

        kept = np.asarray(chunk[keep_flags], dtype=PackedSfenValue)
        if len(kept) > 0:
            with open(output_path, "ab") as f:
                kept.tofile(f)
        kept_total += len(kept)

        print(
            f"progress {end}/{total} keep={kept_total} "
            f"ratio={kept_total / end:.6f}",
            flush=True,
        )

    print(
        f"done input={input_path} output={output_path} total={total} kept={kept_total} "
        f"keep_ratio={kept_total / total:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    app()
