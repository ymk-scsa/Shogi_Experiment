# GNN_Experiment_20251229/train 解説ガイド

このドキュメントは、`train_README.md` とは別に、`train` ディレクトリ内の学習スクリプトを「何をしているか」「どう使い分けるか」の観点で整理したものです。

---

## 1. まず全体像

`train` 配下には、目的の異なる 4 種類のトレーニングスクリプトがあります。

- `train.py`
  - **教師あり学習（棋譜データ）** の基本版
  - 複数モデル（例: `modelA`, `modelB`）を1プロセス内で順次更新できる
- `parallelization_train.py`
  - **8モデル同時学習（modelA〜modelH）** をプロセス並列で実行
  - 各モデルが独立プロセスで動く
- `Imitation_train.py`
  - **模倣学習の強化版**
  - DDP（分散学習）と PER（優先度サンプリング）をサポート
- `ppo_per_train.py`
  - **自己対局ベースの強化学習**
  - PPO + PER + 補助タスク + SSL をまとめた本格学習ループ

---

## 2. 各スクリプトの役割と中身

## `train.py`（教師あり学習の基準スクリプト）

### 役割
- HCPE / PSV 形式の教師データを読み、方策と価値を同時学習する。
- `--models modelA:5,modelB:10` のように、モデルごとに目標エポック数を指定可能。

### 学習の中身
- 出力:
  - policy: `log_softmax` 出力を想定（損失は `NLLLoss`）
  - value: `tanh` 出力（`[-1,1]`）を `(value+1)/2` で `[0,1]` に変換して `MSELoss`
- 主要機能:
  - 定期評価（`--eval-interval`）
  - チェックポイント保存
  - `Ctrl+C` 時の安全中断保存
  - 一定時間ごとのキャッシュ保存
  - 損失グラフ `data/loss_<mode>.png` 出力

### 向いている用途
- まず1つのモデルを安定して学習させたいとき
- 実験条件をシンプルに比較したいとき

---

## `parallelization_train.py`（モデル並列）

### 役割
- `modelA`〜`modelH` を **同時に** 学習したいときのスクリプト。
- `torch.multiprocessing.spawn` で8ワーカーを立て、1ワーカー=1モデルで学習。

### 学習の中身
- 各ワーカーで:
  - モデル作成
  - データローダ読み込み
  - `NLLLoss + MSELoss` で更新
  - `--eval-interval` ごとにテスト精度を記録
  - モデルごとのログ/グラフ/チェックポイントを個別出力
- 最適化手法は `AdamW`（`train.py` は `SGD`）

### 向いている用途
- 複数モデルを同じ条件で一気に回したいとき
- GPUが複数あり、壁時計時間を短縮したいとき

---

## `Imitation_train.py`（模倣学習 + DDP/PER）

### 役割
- より実験的な模倣学習スクリプト。
- `--use-per` で優先度サンプリング、複数GPU時は DDP で並列化。

### 学習の中身
- DDP:
  - `world_size > 1` で `DistributedDataParallel` を利用
  - ランク0中心でログ/保存を担当
- PER:
  - `PrioritySampler` が損失に基づきサンプル優先度を更新
  - 高誤差サンプルを再学習しやすくする狙い
- 損失:
  - 基本は `NLLLoss + MSELoss`
  - PER時は `reduction='none'` でサンプル単位損失を扱う

### 向いている用途
- 模倣学習で「学習効率」を上げる実験をしたいとき
- DDPでスケールさせたいとき

---

## `ppo_per_train.py`（自己対局による強化学習）

### 役割
- 棋譜教師ではなく、自己対局データから学習する強化学習スクリプト。
- このディレクトリ内で最も高機能。

### 学習フェーズ
- Phase 1: 自己対局 (`self_play_one_game`) で経験を収集し、PERバッファへ格納
- Phase 2: バッファが閾値を超えたら PPO 更新 (`ppo_update`)
- 更新後、TD誤差で PER 優先度を更新

### 損失構成（合計損失）
- policy: PPO clipped surrogate
- value: Huber loss
- entropy bonus
- auxiliary loss:
  - king_safety / material / mobility / attack / threat / damage
- SSL loss:
  - マスクした盤面復元（`masked_board`）

### 向いている用途
- AlphaZero系に近い学習を回したいとき
- 補助タスク込みで表現学習を強化したいとき

---

## 3. 使い分けの目安（簡易）

- 最初の基準線を作る: `train.py`
- 8モデルを一気に比較: `parallelization_train.py`
- 模倣学習の効率化を試す: `Imitation_train.py`
- 自己対局ベースの本格RL: `ppo_per_train.py`

---

## 4. 出力ファイルの違い（把握しておくと便利）

- `train.py`
  - checkpoint: `weights/`
  - loss graph: `data/loss_<model>.png`
- `parallelization_train.py`
  - checkpoint: `weights_parallel/`
  - log/graph: `data_parallel/`（モデル別）
- `Imitation_train.py`
  - checkpoint: `weights_imitation/`
  - loss graph: `data/imitation_loss_<model>.png`
- `ppo_per_train.py`
  - checkpoint: `weights/checkpoint-<mode>-*.pth`
  - TensorBoard: `runs/<mode>_<timestamp>/`

---

## 5. 補足（読み解き時の注意）

- `train.py` と `Imitation_train.py` では `HcpeDataLoader` の import パスが異なるため、実行環境によっては import 調整が必要な場合があります。
- `ppo_per_train.py` は `game.board` / `search.mcts` の実装前提が強く、未整備時はダミー経験生成にフォールバックする実装になっています。
- まずは `train.py` を基準に動作確認し、その後に並列化・PER・RLへ段階的に進めるのがおすすめです。

python train/parallelization_train.py --train-data ./datas/Suisho_train.hcpe --test-data ./datas/Suisho_test.hcpe --batch 1024 --epochs 1 --eval-interval 50 --checkpoint-dir ./weights_parallel_test --log-dir ./data_parallel_test --batch 256 --epochs 1 --eval-interval 50 --checkpoint-dir ./weights_parallel_test --log-dir ./data_parallel_test