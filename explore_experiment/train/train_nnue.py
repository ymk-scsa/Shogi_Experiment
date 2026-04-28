"""HalfKAv2 + NNUE-Large の価値学習（HCPE）。

方策は現状の NnueLargeModel にヘッドがないため、教師の best move は読み込むが損失には使わない。
将来ポリシーヘッドを足す場合は cross_entropy を追加する。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import signal
import time
from typing import Optional

import torch
import torch.optim as optim
import typer
from typing_extensions import Annotated

from model.nnue_large import NnueLargeModel
from util.dataloader import HcpeNnueDataLoader
from util.directory import ensure_directory_exists
from util.logger import Logger

train_nnue_app = typer.Typer()


@train_nnue_app.command()
def train(
    train_data: Annotated[list[str], typer.Option(help="training HCPE file(s)")],
    test_data: Annotated[str, typer.Option(help="test HCPE file")],
    gpu: Annotated[int, typer.Option("-g", help="GPU ID (-1 for CPU)")] = 0,
    train_cnt: Annotated[int, typer.Option("-e", help="epochs")] = 1,
    batchsize: Annotated[int, typer.Option("-b", help="batch size")] = 1024,
    testbatchsize: Annotated[int, typer.Option(help="test batch size")] = 1024,
    lr: Annotated[float, typer.Option(help="learning rate")] = 0.001,
    checkpoint_base: Annotated[str, typer.Option(help="checkpoint directory")] = "checkpoint/nnue/",
    checkpoint: Annotated[str, typer.Option(help="checkpoint filename pattern")] = "nnue-{epoch:03}.pth",
    resume: Annotated[str, typer.Option("-r", help="resume from filename under checkpoint_base")] = "",
    eval_interval: Annotated[int, typer.Option(help="eval every N steps")] = 100,
    log: Annotated[Optional[str], typer.Option(help="log file")] = None,
    accum_dim: Annotated[int, typer.Option(help="accumulator width")] = 1024,
    hidden1: Annotated[int, typer.Option(help="hidden1")] = 1024,
    hidden2: Annotated[int, typer.Option(help="hidden2")] = 512,
) -> None:
    """Train NNUE-Large value head from HCPE (HalfKAv2 sparse features)."""

    logging = Logger("train_nnue", log_file=log).get_logger()
    logging.info("batchsize=%s lr=%s", batchsize, lr)

    if gpu >= 0:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    model = NnueLargeModel(accum_dim=accum_dim, hidden1=hidden1, hidden2=hidden2)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

    if resume:
        path = checkpoint_base + resume
        logging.info("Loading checkpoint from %s", path)
        checkpoint_data = torch.load(path, map_location=device)
        epoch = int(checkpoint_data["epoch"])
        t = int(checkpoint_data["t"])
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        optimizer.param_groups[0]["lr"] = lr
    else:
        epoch = 0
        t = 0

    logging.info("Reading training data")
    train_dataloader = HcpeNnueDataLoader(train_data, batchsize, device, shuffle=True)
    logging.info("Reading test data")
    test_dataloader = HcpeNnueDataLoader(test_data, testbatchsize, device, shuffle=False)

    logging.info("train positions = %s", len(train_dataloader))
    logging.info("test positions = %s", len(test_dataloader))

    def binary_accuracy(y: torch.Tensor, truth: torch.Tensor) -> float:
        pred = y >= 0
        tr = truth >= 0.5
        return pred.eq(tr).sum().item() / len(tr)

    def save_checkpoint(checkpoint_path: str) -> None:
        path = checkpoint_path.format(epoch=epoch, step=t)
        logging.info("Saving checkpoint to %s", path)
        ensure_directory_exists(path)
        torch.save(
            {
                "epoch": epoch,
                "t": t,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            path,
        )

    cache_time = time.time()

    def handle_sigint(signum: int, frame: object) -> None:
        if checkpoint:
            logging.info("SIGINT: saving checkpoint...")
            save_checkpoint(checkpoint_base + checkpoint)
        exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    for _e in range(train_cnt):
        epoch += 1
        steps_interval = 0
        sum_loss_interval = 0.0
        steps_epoch = 0
        sum_loss_epoch = 0.0

        for flat_indices, offsets, _move_label, target_result in train_dataloader:
            model.train()
            y = model.forward_sparse_batched(flat_indices, offsets)
            loss = bce_with_logits_loss(y, target_result)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t += 1
            steps_interval += 1
            sum_loss_interval += loss.item()

            if t % eval_interval == 0:
                model.eval()
                fi, off, _ml, tr = test_dataloader.sample()
                with torch.no_grad():
                    yv = model.forward_sparse_batched(fi, off)
                    test_loss = bce_with_logits_loss(yv, tr).item()
                    test_acc = binary_accuracy(yv, tr)

                logging.info(
                    "epoch=%s steps=%s train_loss=%.7f test_loss=%.7f test_acc_value=%.7f",
                    epoch,
                    t,
                    sum_loss_interval / max(steps_interval, 1),
                    test_loss,
                    test_acc,
                )

                steps_epoch += steps_interval
                sum_loss_epoch += sum_loss_interval
                steps_interval = 0
                sum_loss_interval = 0.0

            if time.time() - cache_time > 60 * 60:
                save_checkpoint(checkpoint_base + "cache/" + checkpoint)
                cache_time = time.time()

        steps_epoch += steps_interval
        sum_loss_epoch += sum_loss_interval
        logging.info(
            "epoch=%s steps=%s train_loss_avr=%.7f",
            epoch,
            t,
            sum_loss_epoch / max(steps_epoch, 1),
        )
        if checkpoint:
            save_checkpoint(checkpoint_base + checkpoint)


if __name__ == "__main__":
    train_nnue_app()
