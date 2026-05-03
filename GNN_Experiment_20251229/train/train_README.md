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

train/league_train.py の実装が完了しました。

ユーザーの要望通り、Gumbel AlphaZero式（合法手ロジットにGumbelノイズを加算して方策改善・行動選択を行う形式）の自己対局を取り入れ、総当たりリーグを回しながら学習するスクリプトを構築しています。

主な実装内容
Gumbel AlphaZero 自己対局: 各対局で盤面ごとに推論し、合法手に対応するロジットにGumbelノイズを加えます。それによって最も高い値を持った手を指し手として採用し、方策改善のターゲット（pi_target）として正規化した確率分布を保持します。これにより、高速かつ探索要素を含む自己対局データを生成します。

高速なリーグ回しと同時学習: エポックごとに modelA 〜 modelH の8つのモデルをシャッフルし、4ペア（4局）の対局を同時に行います（総当たり形式のリーグ戦ループ）。ここで生成された4局分の棋譜（全局面の入力特徴量、方策ターゲット、勝敗価値ターゲット、補助タスクターゲット）を使用して、直後に8つのモデルすべてを一斉に学習させます。

グラフ・表の自動出力: 定期的に（デフォルトでは5エポックごと、または終了時）に、以下の2つのファイルを出力します。

league_results/losses.png: 各モデル（8個）の Policy Loss、Value Loss、Total Loss の推移を可視化したグラフ。
league_results/league_stats.md: 通算対戦成績と、直近3000局の成績（勝ち、負け、引き分け、勝率）をまとめたマークダウン形式の表。
実行方法
プロジェクトのルートまたは train/ ディレクトリから、以下のコマンドで実行できます（必要に応じて --epochs で回数を指定してください）：

bash
python train/league_train.py --epochs 1000
必要であればすぐに実行してテストすることも可能です。何か修正したいパラメーター（対局時の手数上限、学習率など）や、確認したい部分はありますか？