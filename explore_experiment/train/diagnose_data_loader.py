import sys
from pathlib import Path

# train/ から直接実行したときにプロジェクトルートを import パスに含める
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import math
import random
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import typer
from cshogi import Board, HuffmanCodedPosAndEval, PackedSfenValue, move16_from_psv
from typing_extensions import Annotated

from model.activation_function import resolve_activation_function
from model.model import PolicyValueResNetModel
from shogi.feature import FEATURES_NUM, MOVE_LABELS_NUM, make_move_label, make_result
from util.dataloader import HcpeDataLoader, PsvDataLoader

app = typer.Typer()


def _psv_score_to_value(score: float, turn: int, scale: float) -> float:
    v = 1.0 / (1.0 + math.exp(-float(score) / scale))
    if turn != 0:
        v = 1.0 - v
    return float(v)


def _sample_records(
    data: np.ndarray,
    max_records: int,
    seed: int,
) -> np.ndarray:
    if len(data) <= max_records:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), size=max_records, replace=False)
    return data[idx]


@app.command()
def run(
    data_path: Annotated[str, typer.Option(help="Path to .hcpe or .bin(PSV) file")],
    data_format: Annotated[Literal["hcpe", "psv"], typer.Option(help="Data format")] = "hcpe",
    max_records: Annotated[int, typer.Option(help="How many records to inspect")] = 20000,
    seed: Annotated[int, typer.Option(help="Random seed for sampling")] = 7,
    batchsize: Annotated[int, typer.Option(help="Batch size for loss probe")] = 1024,
    loss_batches: Annotated[int, typer.Option(help="How many random batches for loss probe")] = 3,
    psv_score_scale: Annotated[float, typer.Option(help="PSV score sigmoid scale")] = 600.0,
    checkpoint: Annotated[Optional[str], typer.Option(help="Checkpoint path to probe initial loss")] = None,
    gpu: Annotated[int, typer.Option("-g", help="GPU ID (-1 for CPU)")] = -1,
    blocks: Annotated[int, typer.Option(help="ResNet blocks")] = 20,
    channels: Annotated[int, typer.Option(help="ResNet channels")] = 192,
    activation_function: Annotated[str, typer.Option(help="Activation function name")] = "relu",
) -> None:
    fmt = data_format.strip().lower()
    board = Board()

    # 1) raw record check
    if fmt == "hcpe":
        raw = np.fromfile(data_path, dtype=HuffmanCodedPosAndEval)
    else:
        raw = np.fromfile(data_path, dtype=PackedSfenValue)

    if len(raw) == 0:
        raise ValueError(f"empty data: {data_path}")

    sampled = _sample_records(raw, max_records=max_records, seed=seed)
    legal_ok_by_move = 0
    legal_ok_by_label = 0
    label_oob = 0
    value_list: list[float] = []
    turn_black = 0
    turn_white = 0

    for rec in sampled:
        if fmt == "hcpe":
            board.set_hcp(rec["hcp"])
            move16 = int(rec["bestMove16"])
            v = float(make_result(int(rec["gameResult"]), board.turn))
        else:
            board.set_psfen(rec["sfen"])
            move16 = int(move16_from_psv(int(rec["move"])))
            v = _psv_score_to_value(float(rec["score"]), board.turn, psv_score_scale)

        if board.turn == 0:
            turn_black += 1
        else:
            turn_white += 1

        label = make_move_label(move16, board.turn)
        if not (0 <= label < MOVE_LABELS_NUM):
            label_oob += 1
        legal_moves = list(board.legal_moves)
        legal_labels = {make_move_label(m, board.turn) for m in legal_moves}
        if move16 in set(legal_moves):
            legal_ok_by_move += 1
        if label in legal_labels:
            legal_ok_by_label += 1

        value_list.append(v)

    v_arr = np.asarray(value_list, dtype=np.float64)
    print("=== Data Integrity ===")
    print(f"format={fmt} total={len(raw)} sampled={len(sampled)}")
    print(
        f"legal_move_ratio(raw_move_compare)={legal_ok_by_move / len(sampled):.6f} "
        f"({legal_ok_by_move}/{len(sampled)})"
    )
    print(
        f"legal_move_ratio(label_compare)={legal_ok_by_label / len(sampled):.6f} "
        f"({legal_ok_by_label}/{len(sampled)})"
    )
    print(f"label_out_of_bounds={label_oob}")
    print(f"turn_distribution black={turn_black} white={turn_white}")
    print(
        "value_stats "
        f"min={float(v_arr.min()):.6f} max={float(v_arr.max()):.6f} "
        f"mean={float(v_arr.mean()):.6f} std={float(v_arr.std()):.6f}"
    )
    print(
        "value_bins "
        f"[0,0.1]={int(np.sum((v_arr >= 0.0) & (v_arr < 0.1)))} "
        f"[0.1,0.9]={int(np.sum((v_arr >= 0.1) & (v_arr <= 0.9)))} "
        f"(0.9,1]={int(np.sum((v_arr > 0.9) & (v_arr <= 1.0)))}"
    )

    # 2) dataloader -> checkpoint loss probe
    if checkpoint:
        device = torch.device(f"cuda:{gpu}") if gpu >= 0 else torch.device("cpu")
        if fmt == "hcpe":
            loader = HcpeDataLoader(data_path, batchsize, device, shuffle=True, features_num=FEATURES_NUM)
        else:
            loader = PsvDataLoader(
                data_path,
                batchsize,
                device,
                shuffle=True,
                features_num=FEATURES_NUM,
                score_scale=psv_score_scale,
            )

        model = PolicyValueResNetModel(
            blocks=blocks,
            channels=channels,
            input_features=FEATURES_NUM,
            activation_function=resolve_activation_function(activation_function),
        ).to(device)
        cp = torch.load(checkpoint, map_location=device)
        state = cp["model"] if isinstance(cp, dict) and "model" in cp else cp
        model.load_state_dict(state)
        model.eval()

        ce = torch.nn.CrossEntropyLoss()
        bce = torch.nn.BCEWithLogitsLoss()
        losses_p = []
        losses_v = []

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            for _ in range(loss_batches):
                x, move_label, result = loader.sample()
                y1, y2 = model(x)
                losses_p.append(float(ce(y1, move_label).item()))
                losses_v.append(float(bce(y2, result).item()))

        print("=== Checkpoint Loss Probe ===")
        print(
            f"policy_loss mean={np.mean(losses_p):.6f} std={np.std(losses_p):.6f} "
            f"(batches={loss_batches})"
        )
        print(
            f"value_loss  mean={np.mean(losses_v):.6f} std={np.std(losses_v):.6f} "
            f"(batches={loss_batches})"
        )


if __name__ == "__main__":
    app()
