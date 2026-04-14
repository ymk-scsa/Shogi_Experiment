import sys
import os
import time
import argparse
import logging
import signal
import matplotlib.pyplot as plt  # 追加
from collections import defaultdict # 追加

import torch
import torch.optim as optim

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model import create_model
from game.board import FEATURES_SETTINGS, MOVE_LABELS_NUM
from data.past_buffer import HcpeDataLoader, PsvDataLoader


# ===== ロガー設定 =====
def _get_logger(log_file: str = None) -> logging.Logger:
    logger = logging.getLogger("train")
    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ===== 精度計算 =====

def accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    """方策の正解率（top-1）。"""
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)


def binary_accuracy(y: torch.Tensor, t: torch.Tensor) -> float:
    """
    価値の二値正解率。
    y: model の Tanh 出力 ([-1, 1]) → (y+1)/2 で [0,1] に変換して比較。
    t: 教師信号 [0, 1]
    """
    pred  = ((y + 1.0) / 2.0) >= 0.5
    truth = t >= 0.5
    return pred.eq(truth).sum().item() / len(t)


# ===== チェックポイント =====

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d)


def save_checkpoint(path: str, model, optimizer, epoch: int, step: int) -> None:
    _ensure_dir(path)
    torch.save(
        {
            "epoch":     epoch,
            "t":         step,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


# ===== 評価損失計算（バリデーション用）=====

def evaluate(model, dataloader, device) -> dict:
    """
    テストデータでモデルを評価して損失・正解率を返す。
    policy: CrossEntropyLoss / value: MSELoss
    """
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    model.eval()
    total_pl, total_vl, total_pa, total_va, steps = 0.0, 0.0, 0.0, 0.0, 0
    # loader が空の場合の対策
    if len(dataloader) == 0:
        return {
            "policy_loss": 0.0, "value_loss": 0.0,
            "policy_accuracy": 0.0, "value_accuracy": 0.0
        }

    with torch.no_grad():
        for x, move_label, result in dataloader:
            policy_logits, value = model(x, return_aux=False)
            pl = ce_loss(policy_logits, move_label).item()
            vl = mse_loss((value + 1.0) / 2.0, result).item()
            total_pl += pl
            total_vl += vl
            total_pa += accuracy(policy_logits, move_label)
            total_va += binary_accuracy(value, result)
            steps += 1
    return {
        "policy_loss":    total_pl / max(steps, 1),
        "value_loss":     total_vl / max(steps, 1),
        "policy_accuracy": total_pa / max(steps, 1),
        "value_accuracy":  total_va / max(steps, 1),
    }

def save_loss_graph(history, output_dir="data/"):
    _ensure_dir(output_dir)
    for mode, data in history.items():
        if not data["step"]: continue
        plt.figure(figsize=(10, 6))
        plt.plot(data["step"], data["p_loss"], label="Policy Loss")
        plt.plot(data["step"], data["v_loss"], label="Value Loss")
        plt.plot(data["step"], data["t_loss"], label="Total Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss - {mode}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"loss_{mode}.png"))
        plt.close()

# ===== メイン学習関数 =====

def train(
    train_data:       list,
    test_data:        str,
    model_configs:    list,  # [(mode, target_epochs), ...]
    gpu:              int   = 0,
    batchsize:        int   = 1024,
    testbatchsize:    int   = 1024,
    lr:               float = 0.001,
    checkpoint_dir:   str   = "weights/",
    checkpoint_name:  str   = "checkpoint-{model}-{epoch:03}.pth",
    resume:           str   = "",
    eval_interval:    int   = 100,
    log_file:         str   = None,
    input_features:   int   = 0,
    data_format:      str   = "hcpe",
) -> None:
    """
    教師あり学習のメインループ。

    Args:
        train_data:      学習データファイルパスのリスト（HCPE形式）
        test_data:       テストデータファイルパス
        gpu:             GPU ID（-1 で CPU）
        train_cnt:       エポック数
        batchsize:       ミニバッチサイズ
        testbatchsize:   テスト時のバッチサイズ
        lr:              学習率
        checkpoint_dir:  チェックポイント保存先ディレクトリ
        checkpoint_name: チェックポイントファイル名（{epoch}, {step} が使える）
        resume:          再開するチェックポイントファイル名（空で新規学習）
        eval_interval:   評価を行うステップ間隔
        log_file:        ログファイルパス
        input_features:  特徴量モード (0: default, 1: kiki, 2: himo, 3: small)
        blocks_mode:     モデルブロック構成 ("30blocks" / "40blocks")
    """
    logger = _get_logger(log_file)
    logger.info("batchsize=%d  lr=%f  models=%s", batchsize, lr, str(model_configs))
    
    # --- 追加：安全停止用のフラグ ---
    stop_requested = False

    def handler(signum, frame):
        nonlocal stop_requested
        logger.info("\n[Interrupt] 中断信号を受け取りました。現在のステップで保存して終了します...")
        stop_requested = True

    # Ctrl+C (SIGINT) を捕捉するように設定
    signal.signal(signal.SIGINT, handler)
    # ------------------------------

    # 特徴量設定
    features_setting = FEATURES_SETTINGS[input_features]
    input_ch = features_setting.features_num
    logger.info("input_channels=%d", input_ch)
    
    # 損失関数の定義（policy は生logitsを返す想定）
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    # --- 修正箇所：マルチGPU対応 ---
    num_gpus = torch.cuda.device_count() if gpu >= 0 else 0
    device = torch.device(f"cuda:{gpu}") if gpu >= 0 else torch.device("cpu")
    logger.info("device: %s (Available GPUs: %d)", device, num_gpus)

    # グラフ用ログ保存変数
    history = defaultdict(lambda: {"step": [], "p_loss": [], "v_loss": [], "t_loss": []})

    # モデル・オプティマイザの初期化
    models = []
    max_epochs = 0
    
    for mode, target_epoch in model_configs:
        model = create_model(input_channels=input_ch, num_actions=MOVE_LABELS_NUM, mode=mode)
        model.to(device)
        
        # GPUが2枚以上あればマルチGPU化
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
            logger.info("[%s] Multi-GPU mode enabled (using %d GPUs)", mode, num_gpus)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        
        models.append({
            "model": model, "optimizer": optimizer, "mode": mode,
            "target_epoch": target_epoch, "current_epoch": 0,
            "current_step": 0, "done": False
        })
        max_epochs = max(max_epochs, target_epoch)

    # --- 修正：再開機能（DataParallel対応版） ---
    if resume:
        resume_path = os.path.join(checkpoint_dir, resume)
        if os.path.exists(resume_path):
            logger.info("Resume from %s", resume_path)
            ckpt = torch.load(resume_path, map_location=device)
            for m_info in models:
                m_info["current_epoch"] = ckpt.get("epoch", 0)
                m_info["current_step"] = ckpt.get("t", 0)
                
                state_dict = ckpt["model"]
                # DataParallelの有無によるキーのズレを自動補正
                new_state_dict = {}
                is_model_dp = isinstance(m_info["model"], torch.nn.DataParallel)
                for k, v in state_dict.items():
                    name = k
                    if is_model_dp and not k.startswith('module.'):
                        name = 'module.' + k # 追加
                    elif not is_model_dp and k.startswith('module.'):
                        name = k[7:] # 削除
                    new_state_dict[name] = v
                
                m_info["model"].load_state_dict(new_state_dict)
                m_info["optimizer"].load_state_dict(ckpt["optimizer"])
        else:
            logger.warning("Resume file not found: %s. Starting from scratch.", resume_path)

    # データローダー
    logger.info("data_format=%s", data_format)
    if data_format == "psv":
        LoaderClass = PsvDataLoader
    else:
        LoaderClass = HcpeDataLoader

    logger.info("Loading training data ...")
    train_loader = LoaderClass(
        train_data, batchsize, device, shuffle=True, features_mode=input_features
    )
    logger.info("Loading test data ...")
    test_loader = LoaderClass(
        test_data, testbatchsize, device, features_mode=input_features
    )
    logger.info("train=%d positions  test=%d positions", len(train_loader), len(test_loader))

    # キャッシュ保存タイマー
    cache_time = time.time()

    # ===== 学習ループ =====
    for e in range(max_epochs):
        epoch_idx = e + 1
        
        # 各モデルの進捗管理
        epoch_stats = []
        for m_idx, m_info in enumerate(models):
            if epoch_idx > m_info["target_epoch"]:
                m_info["done"] = True
                epoch_stats.append(None)
                continue
            
            m_info["current_epoch"] = epoch_idx
            epoch_stats.append({
                "steps_interval": 0, "sum_pl_interval": 0.0, "sum_vl_interval": 0.0,
                "steps_epoch": 0, "sum_pl_epoch": 0.0, "sum_vl_epoch": 0.0
            })

        # 全モデルが終了していたら抜ける
        if all(m["done"] for m in models):
            break

        logger.info("--- Starting Epoch %d ---", epoch_idx)

        for x, move_label, result in train_loader:
            for m_idx, m_info in enumerate(models):
                if m_info["done"]:
                    continue

                model = m_info["model"]
                optimizer = m_info["optimizer"]
                stats = epoch_stats[m_idx]

                model.train()
                policy_logits, value = model(x, return_aux=False)
                loss_policy = ce_loss(policy_logits, move_label)
                value_01 = (value + 1.0) / 2.0
                loss_value = mse_loss(value_01, result)
                loss = loss_policy + loss_value

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m_info["current_step"] += 1
                stats["steps_interval"] += 1
                stats["sum_pl_interval"] += loss_policy.item()
                stats["sum_vl_interval"] += loss_value.item()

                # 評価間隔
                if m_info["current_step"] % eval_interval == 0:
                    model.eval()
                    x_t, ml_t, res_t = test_loader.sample()
                    with torch.no_grad():
                        p_t, v_t = model(x_t, return_aux=False)
                        test_pl = ce_loss(p_t, ml_t).item()
                        test_vl = mse_loss((v_t + 1.0) / 2.0, res_t).item()
                        test_pa = accuracy(p_t, ml_t)
                        test_va = binary_accuracy(v_t, res_t)

                    logger.info(
                        "[%s] epoch=%d step=%d train_loss(p/v/t)=%.4f/%.4f/%.4f test_acc(p/v)=%.4f/%.4f",
                        m_info["mode"], epoch_idx, m_info["current_step"],
                        stats["sum_pl_interval"] / stats["steps_interval"],
                        stats["sum_vl_interval"] / stats["steps_interval"],
                        (stats["sum_pl_interval"] + stats["sum_vl_interval"]) / stats["steps_interval"],
                        test_pa, test_va
                    )
                    
                    # --- 追加：グラフ描画用データの保存 ---
                    h = history[m_info["mode"]]
                    avg_p_loss = stats["sum_pl_interval"] / stats["steps_interval"]
                    avg_v_loss = stats["sum_vl_interval"] / stats["steps_interval"]

                    h["step"].append(m_info["current_step"])
                    h["p_loss"].append(avg_p_loss)
                    h["v_loss"].append(avg_v_loss)
                    h["t_loss"].append(avg_p_loss + avg_v_loss)

                    # 画像ファイルを更新（data/ フォルダに出力）
                    save_loss_graph(history, output_dir="data/")
                    # ------------------------------------

                    stats["steps_epoch"] += stats["steps_interval"]
                    stats["sum_pl_epoch"] += stats["sum_pl_interval"]
                    stats["sum_vl_epoch"] += stats["sum_vl_interval"]
                    stats["steps_interval"] = 0
                    stats["sum_pl_interval"] = 0.0
                    stats["sum_vl_interval"] = 0.0

            # --- 修正：安全停止（Ctrl+C）または 1時間キャッシュ ---
            if stop_requested or (time.time() - cache_time > 3600):
                for m_info in models:
                    if m_info["done"]: continue
                    # ファイル名に "interrupted" や "cache" を含めて保存
                    suffix = "interrupted" if stop_requested else "cache"
                    cp_name = f"checkpoint-{m_info['mode']}-step{m_info['current_step']}-{suffix}.pth"
                    save_checkpoint(os.path.join(checkpoint_dir, cp_name), m_info["model"], m_info["optimizer"], m_info["current_epoch"], m_info["current_step"])
                    logger.info("[%s] 中断用チェックポイントを保存しました: %s", m_info["mode"], cp_name)
                
                if stop_requested:
                    logger.info("学習を正常に中断しました。お疲れ様でした。")
                    sys.exit(0) # プログラムを終了
                
                cache_time = time.time() # キャッシュ時間をリセット

        # エポック終了
        for m_idx, m_info in enumerate(models):
            if m_info["done"]: continue
            
            optimizer = m_info["optimizer"]
            stats = epoch_stats[m_idx]
            
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.95
            
            stats["steps_epoch"] += stats["steps_interval"]
            stats["sum_pl_epoch"] += stats["sum_pl_interval"]
            stats["sum_vl_epoch"] += stats["sum_vl_interval"]

            if stats["steps_epoch"] > 0:
                eval_res = evaluate(m_info["model"], test_loader, device)
                logger.info(
                    "=== [%s] Epoch %d done (step=%d) train_loss_avg=%.6f test_acc(p/v)=%.4f/%.4f ===",
                    m_info["mode"], epoch_idx, m_info["current_step"],
                    (stats["sum_pl_epoch"] + stats["sum_vl_epoch"]) / stats["steps_epoch"],
                    eval_res["policy_accuracy"], eval_res["value_accuracy"]
                )

            # 保存
            cp_name = checkpoint_name.format(model=m_info["mode"], epoch=epoch_idx, step=m_info["current_step"])
            save_checkpoint(os.path.join(checkpoint_dir, cp_name), m_info["model"], m_info["optimizer"], epoch_idx, m_info["current_step"])
            logger.info("[%s] Checkpoint saved: %s", m_info["mode"], cp_name)


# ===== CLI エントリーポイント =====

def _parse_args():
    parser = argparse.ArgumentParser(description="GNN_Experiment_20251229 教師あり学習スクリプト")
    parser.add_argument("--train-data",     nargs="+", required=True,   help="学習データ (HCPE) ファイル")
    parser.add_argument("--test-data",      required=True,               help="テストデータ (HCPE) ファイル")
    parser.add_argument("--models",         default="modelA:1",          help="モデル定義 (例: modelA:5,modelB:10)")
    parser.add_argument("-g", "--gpu",      type=int,   default=0,       help="GPU ID (-1 = CPU)")
    parser.add_argument("-b", "--batch",    type=int,   default=1024,    help="ミニバッチサイズ")
    parser.add_argument("--test-batch",     type=int,   default=1024,    help="テスト時バッチサイズ")
    parser.add_argument("--lr",             type=float, default=0.001,   help="学習率")
    parser.add_argument("--checkpoint-dir", default="weights/",          help="チェックポイント保存先")
    parser.add_argument("--checkpoint",     default="checkpoint-{model}-{epoch:03}.pth", help="チェックポイントファイル名テンプレート")
    parser.add_argument("-r", "--resume",   default="",                  help="再開チェックポイントファイル名 (単一モデル時のみ)")
    parser.add_argument("--eval-interval",  type=int,   default=100,     help="評価間隔（ステップ）")
    parser.add_argument("--log",            default=None,                help="ログファイルパス")
    parser.add_argument("-i", "--input-features", type=int, default=0,  help="特徴量モード (0:default 1:kiki 2:himo 3:small)")
    parser.add_argument("--format",          default="hcpe",
                        choices=["hcpe", "psv"],                          help="教師データ形式 (hcpe / psv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    
    # モデル設定のパース
    model_configs = []
    for m_spec in args.models.split(","):
        parts = m_spec.split(":")
        mode = parts[0]
        epochs = int(parts[1]) if len(parts) > 1 else 1
        model_configs.append((mode, epochs))

    train(
        train_data=args.train_data,
        test_data=args.test_data,
        model_configs=model_configs,
        gpu=args.gpu,
        batchsize=args.batch,
        testbatchsize=args.test_batch,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint,
        resume=args.resume,
        eval_interval=args.eval_interval,
        log_file=args.log,
        input_features=args.input_features,
        data_format=args.format,
    )
