"""
parallel_train.py
=================

modelA〜modelH の 8 モデルを、それぞれ独立ワーカープロセスで並列学習するスクリプト。

【変更点】
- ShogiDataLoader に aux_labels=True を渡し、補助タスクのラベルを受け取る
- 補助タスク損失を合算して学習（ラベルは board から自動計算、人間は重み λ を設定するだけ）
- aux 損失の重みは小さめ (0.1〜0.2) で policy/value の学習を邪魔しない設計

損失関数の対応:
  king_safety  : MSELoss
  material     : MSELoss
  mobility     : MSELoss
  attack/threat/damage : BCEWithLogitsLoss（モデル出力は sigmoid なし raw logits）
"""

import sys
import os
import argparse
import logging
import signal
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import create_model
from game.board import FEATURES_SETTINGS, MOVE_LABELS_NUM
from data.buffer import ShogiDataLoader


# ---------------------------------------------------------------------------
# ロガー
# ---------------------------------------------------------------------------

def _get_logger(log_file: str = None, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        fmt = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        ch  = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 精度計算
# ---------------------------------------------------------------------------

def accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    """Policy top-1 正解率"""
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)


def binary_accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    """Value 二値正解率 (tanh→[-1,1] を [0,1] に変換して閾値0.5で判定)"""
    pred  = ((y + 1.0) / 2.0) >= 0.5
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)


# ---------------------------------------------------------------------------
# ファイル保存
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d)


def save_checkpoint(path, model, optimizer, epoch, step):
    _ensure_dir(path)
    torch.save(
        {"epoch": epoch, "t": step,
         "model": model.state_dict(),
         "optimizer": optimizer.state_dict()},
        path,
    )


def save_loss_graph(history, mode, output_dir="data/"):
    if not HAS_MATPLOTLIB:
        return
    _ensure_dir(output_dir)
    data = history[mode]
    if not data["step"]:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(data["step"], data["p_loss"],   label="Policy Loss")
    plt.plot(data["step"], data["v_loss"],   label="Value Loss")
    plt.plot(data["step"], data["aux_loss"], label="Aux Loss")
    plt.plot(data["step"], data["t_loss"],   label="Total Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss — {mode}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_{mode}.png"))
    plt.close()


# ---------------------------------------------------------------------------
# 補助タスク損失の計算
# ---------------------------------------------------------------------------

# ✅ 推奨: モジュールレベルで一度だけ作成
_MSE_LOSS = nn.MSELoss()
_BCE_LOSS = nn.BCEWithLogitsLoss()

def compute_aux_loss(
    aux_pred: dict,
    aux_lbl:  dict,
    device:   torch.device,
    w_scalar: float = 0.1,
    w_map:    float = 0.2,
) -> torch.Tensor:
    """
    aux head の出力と教師ラベルから補助タスク損失を計算する。
    ...（元のdocstringをそのまま）
    """
    loss = torch.tensor(0.0, device=device)
    loss = loss + w_scalar * _MSE_LOSS(aux_pred["king_safety"], aux_lbl["king_safety"])
    loss = loss + w_scalar * _MSE_LOSS(aux_pred["material"],    aux_lbl["material"])
    loss = loss + w_scalar * _MSE_LOSS(aux_pred["mobility"],    aux_lbl["mobility"])
    loss = loss + w_map    * _BCE_LOSS(aux_pred["attack"],  aux_lbl["attack"])
    loss = loss + w_map    * _BCE_LOSS(aux_pred["threat"],  aux_lbl["threat"])
    loss = loss + w_map    * _BCE_LOSS(aux_pred["damage"],  aux_lbl["damage"])
    return loss

"""
    # ---- スカラー系 (MSE) ----
    # king_safety: モデル出力 (B,2)、教師 (B,2)
    loss = loss + w_scalar * mse(aux_pred["king_safety"], aux_lbl["king_safety"])
    # material: モデル出力 (B,1)、教師 (B,1)
    loss = loss + w_scalar * mse(aux_pred["material"], aux_lbl["material"])
    # mobility: モデル出力 (B,1)、教師 (B,1)
    loss = loss + w_scalar * mse(aux_pred["mobility"], aux_lbl["mobility"])

    # ---- マップ系 (BCEWithLogits) ----
    # モデル出力は raw logits (sigmoid なし)、教師は {0,1} バイナリ
    # attack: (B,1,9,9)
    loss = loss + w_map * bce(aux_pred["attack"], aux_lbl["attack"])
    # threat: (B,1,9,9)
    loss = loss + w_map * bce(aux_pred["threat"], aux_lbl["threat"])
    # damage: (B,1,9,9)
    loss = loss + w_map * bce(aux_pred["damage"], aux_lbl["damage"])

    return loss
"""

# ---------------------------------------------------------------------------
# ワーカープロセス
# ---------------------------------------------------------------------------

def train_worker(rank: int, num_gpus: int, args: argparse.Namespace, model_list: List[str]):
    model_mode = model_list[rank]

    # デバイス割り当て
    if num_gpus > 0:
        device = torch.device(f"cuda:{rank % num_gpus}")
    else:
        device = torch.device("cpu")

    # ログ
    log_suffix = f"_{args.run_tag}" if args.run_tag else ""
    log_file   = os.path.join(args.log_dir, f"train_{model_mode}{log_suffix}.log")
    _ensure_dir(log_file)
    logger = _get_logger(log_file, name=model_mode)
    logger.info("Starting worker for %s on %s", model_mode, device)

    # 特徴量設定
    features_setting = FEATURES_SETTINGS[args.input_features]
    input_ch         = features_setting.features_num

    # モデル
    model = create_model(input_channels=input_ch, num_actions=MOVE_LABELS_NUM, mode=model_mode)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # データローダー
    # aux_labels=True にすることで補助タスクラベルも自動生成される
    logger.info("Loading training data (format=%s, aux=%s)...", args.format, args.use_aux)
    train_loader = ShogiDataLoader(
        args.train_data, args.batch, device,
        format=args.format, shuffle=True,
        features_mode=args.input_features,
        aux_labels=args.use_aux,
    )
    test_loader = ShogiDataLoader(
        args.test_data, args.test_batch, device,
        format=args.format, shuffle=False,
        features_mode=args.input_features,
        aux_labels=False,   # バリデーションは policy/value のみ
    )

    # チェックポイント再開
    current_step = 0
    start_epoch  = 0
    if args.resume:
        resume_file = args.resume.replace("{model}", model_mode)
        if os.path.exists(resume_file):
            ckpt = torch.load(resume_file, map_location=device, weights_only=True)
            start_epoch  = ckpt.get("epoch", 0)
            current_step = ckpt.get("t", 0)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            logger.info("Resumed from %s", resume_file)

    history = defaultdict(lambda: {
        "step": [], "p_loss": [], "v_loss": [], "aux_loss": [], "t_loss": []
    })

    # Ctrl+C ハンドラ
    stop_requested = False
    def handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, handler)

    # 損失関数
    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    # ===== 学習ループ =====
    for epoch in range(start_epoch, args.epochs):
        epoch_idx = epoch + 1
        model.train()

        running_p   = 0.0
        running_v   = 0.0
        running_aux = 0.0
        steps_intvl = 0

        logger.info("--- Epoch %d ---", epoch_idx)

        for batch in train_loader:
            if stop_requested:
                break

            if args.use_aux:
                # aux_labels=True のとき 4要素タプル
                x, move_label, result, aux_lbl = batch
                policy_logits, value, aux_pred = model(x, return_aux=True)

                loss_p   = ce_loss(policy_logits, move_label)
                loss_v   = mse_loss((value + 1.0) / 2.0, result)
                loss_aux = compute_aux_loss(
                    aux_pred, aux_lbl, device,
                    w_scalar=args.aux_w_scalar,
                    w_map=args.aux_w_map,
                )
                loss = loss_p + loss_v + loss_aux
            else:
                x, move_label, result = batch
                policy_logits, value  = model(x, return_aux=False)
                loss_p   = ce_loss(policy_logits, move_label)
                loss_v   = mse_loss((value + 1.0) / 2.0, result)
                loss_aux = torch.tensor(0.0)
                loss     = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_step += 1
            steps_intvl  += 1
            running_p    += loss_p.item()
            running_v    += loss_v.item()
            running_aux  += loss_aux.item()

            # 評価ステップ
            if current_step % args.eval_interval == 0:
                n = max(1, steps_intvl)
                avg_p, avg_v, avg_aux = running_p/n, running_v/n, running_aux/n

                model.eval()
                pa_sum = va_sum = 0.0
                with torch.no_grad():
                    for _ in range(args.eval_batches):
                        tx, tml, tres = test_loader.sample()
                        p_t, v_t      = model(tx, return_aux=False)
                        pa_sum += accuracy(p_t, tml)
                        va_sum += binary_accuracy(v_t, tres)
                test_pa = pa_sum / max(1, args.eval_batches)
                test_va = va_sum / max(1, args.eval_batches)

                logger.info(
                    "Epoch %d Step %d | Loss P/V/Aux: %.4f/%.4f/%.4f | "
                    "Test Acc P/V: %.4f/%.4f",
                    epoch_idx, current_step,
                    avg_p, avg_v, avg_aux,
                    test_pa, test_va,
                )

                history[model_mode]["step"].append(current_step)
                history[model_mode]["p_loss"].append(avg_p)
                history[model_mode]["v_loss"].append(avg_v)
                history[model_mode]["aux_loss"].append(avg_aux)
                history[model_mode]["t_loss"].append(avg_p + avg_v + avg_aux)
                save_loss_graph(history, model_mode, output_dir=args.log_dir)

                running_p = running_v = running_aux = 0.0
                steps_intvl = 0
                model.train()

        # エポック終了: checkpoint 保存
        cp_suffix = f"-{args.run_tag}" if args.run_tag else ""
        cp_name   = f"parallel-{model_mode}-ep{epoch_idx:03}{cp_suffix}.pth"
        save_checkpoint(
            os.path.join(args.checkpoint_dir, cp_name),
            model, optimizer, epoch_idx, current_step,
        )
        logger.info("Epoch %d finished. Checkpoint: %s", epoch_idx, cp_name)

        if stop_requested:
            logger.info("Interrupted. Saving state...")
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"interrupted-{model_mode}{cp_suffix}.pth"),
                model, optimizer, epoch_idx, current_step,
            )
            break


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel training for modelA-modelH")
    parser.add_argument("--train-data",      nargs="+", required=True)
    parser.add_argument("--test-data",       required=True)
    parser.add_argument("--batch",           type=int,   default=1024,
                        choices=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--test-batch",      type=int,   default=1024)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir",  default="weights_parallel/")
    parser.add_argument("--log-dir",         default="data_parallel/")
    parser.add_argument("--eval-interval",   type=int,   default=100)
    parser.add_argument("--eval-batches",    type=int,   default=4)
    parser.add_argument("--input-features",  type=int,   default=0)
    parser.add_argument("--format",          default="hcpe", choices=["hcpe", "psv"])
    parser.add_argument("--resume",          default="")
    parser.add_argument("--models",
                        default="modelA,modelB,modelC,modelD,modelE,modelF,modelG,modelH")
    parser.add_argument("--run-tag",         default="")

    # aux 関連
    parser.add_argument("--use-aux",         action="store_true",
                        help="補助タスクの学習を有効化する")
    parser.add_argument("--aux-w-scalar",    type=float, default=0.1,
                        help="king_safety/material/mobility の損失重み (default: 0.1)")
    parser.add_argument("--aux-w-map",       type=float, default=0.2,
                        help="attack/threat/damage マップの損失重み (default: 0.2)")

    args = parser.parse_args()

    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")

    if not args.run_tag:
        args.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_list  = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        raise ValueError("No models specified via --models.")
    num_workers = len(model_list)

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s). Spawning {num_workers} worker(s).")
    print(f"Auxiliary loss: {'ENABLED' if args.use_aux else 'DISABLED'}")
    if args.use_aux:
        print(f"  w_scalar={args.aux_w_scalar}, w_map={args.aux_w_map}")

    _ensure_dir(args.checkpoint_dir)
    _ensure_dir(args.log_dir)

    mp.spawn(train_worker, nprocs=num_workers, args=(num_gpus, args, model_list))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
