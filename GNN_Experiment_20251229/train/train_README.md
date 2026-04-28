# GNN Shogi Training フォルダ概要

このディレクトリには、将棋 AI モデル（HybridAlphaZeroNet）の学習用スクリプトが格納されています。
現状、イミテーション学習（教師あり学習）と強化学習（自己対局）の2つの主要なパスが整備されています。

## ディレクトリ構成
- `data/buffer.py`: データ読み込み（PSV/HCPE）および経験再生バッファの統合モジュール。
- `train/parallel_train.py`: 8種類のモデル（modelA〜modelH）を並列に教師あり学習するスクリプト。
- `train/reinforcement_train.py`: Gumbel AlphaZero 方式による強化学習スクリプト。
- `selfplay_data/`: 自己対局で生成されたデータの格納先。

---

## 1. 教師あり学習 (parallel_train.py)

`Suisho10Mn_psv.bin` などの教師データを用いて、複数のモデル構成（modelA〜modelH）を同時に学習します。
各モデルは独立したプロセスで実行され、利用可能な GPU に自動的に割り振られます。

### 実行例
```powershell
# 教師データ (PSV形式) を指定して8モデルを並列学習
python train/parallel_train.py `
    --train-data data/Suisho10Mn_psv.bin `
    --test-data data/test.psv `
    --format psv `
    --batch 1024 `
    --epochs 50 `
    --checkpoint-dir weights/parallel/
```

### 主な引数
- `--train-data`: 学習用棋譜ファイル（複数指定可）。
- `--format`: データ形式 (`psv` または `hcpe`)。`Suisho10Mn_psv.bin` を使う場合は `psv` を指定。
- `--models`: 学習対象のモデル名（カンマ区切り）。デフォルトは `modelA,modelB,...,modelH`。
- `--batch`: 1ワーカーあたりのバッチサイズ。

---

## 2. 強化学習 (reinforcement_train.py)

自己対局によって生成されたデータを用いて、モデルを継続的に強化します。
Gumbel AlphaZero 方式に基づき、バッファから優先度付きサンプリングを行って学習します。

### 実行例
```powershell
# selfplay_data フォルダ内のデータを使用して学習を開始
python train/reinforcement_train.py `
    --mode modelD `
    --batch 512 `
    --lr 0.0002 `
    --steps_per_iter 100
```

### ワークフロー
1. 自己対局プログラム（別プロセス）が `selfplay_data/` に棋譜データを書き出す。
2. `reinforcement_train.py` がそれらのデータを読み込み、バッファを更新する。
3. バッファからデータをサンプリングし、方策（Policy）と価値（Value）を学習する。

---

## 3. データローダーとバッファ (data/buffer.py)

`buffer.py` は以下の2つの役割を果たします。

- **イミテーション学習時**: `ShogiDataLoader` クラスが PSV/HCPE ファイルを高速に読み込み、盤面特徴量と教師ラベル（指し手・評価値）を生成します。
- **強化学習時**: `PrioritizedReplayBuffer` クラスが過去の経験を保持し、損失（TD誤差）の大きいデータを優先的にサンプリングします。

---

## 注意事項
- **メモリ消費**: 8モデルを並列学習する場合、モデルごとにデータがロードされるため、CPUメモリ (RAM) の消費量に注意してください。
- **チェックポイント**: 学習された重みは `--checkpoint-dir` で指定したディレクトリにモデル名ごとに保存されます。