## train.py
```
棋譜データ（HCPEフォーマット）を用いた教師あり学習スクリプト。
shogiAI の train.py を GNN_Experiment_20251229 の HybridAlphaZeroNet 用に移植。

損失関数:
  - policy: NLLLoss（HybridAlphaZeroNet の出力が log_softmax 済みのため）
  - value:  MSELoss（Tanh 出力 [-1, 1] を [0, 1] に変換してから計算）
```

具体的な実行手順
```
# Shogi_Experience フォルダに移動してから実行
cd c:\Users\tkksn\Desktop\Shogi_Experience\Shogi_Experience\GNN_Experiment_20251229

python train/train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:1 ^
  --batch 1024 ^
  --gpu 0
```

1つのモデルを学習するコマンド
```
python train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:1 ^
  --batch 1024 ^
  --gpu 0 ^
  --log train_modelA.log
```
複数のモデルを同時に学習するコマンド
```
python train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:5,modelB:10 ^
  --batch 512 ^
  --gpu 0
```

中断から再開するコマンド
```
#python train.py ... --resume checkpoint-modelA.pth
python train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:5 ^
  --resume checkpoint-modelA-step2500-interrupted.pth
```


8個のモデルを同時に学習するコマンド
```
.\venv\Scripts\python.exe train\parallelization_train.py `
    --train-data data/train.hcpe `
    --test-data data/test.hcpe `
    --batch 1024 `
    --epochs 10
```


## parallelization_train.py の実装と修正
8つのモデル（modelA 〜 modelH）を同時に、かつ並列で学習できるように parallelization_train.py を実装しました。また、プロジェクト内のインポートエラーを修正し、スクリプトが正常に動作するように調整しました。

実施内容

```
1. parallelization_train.py の新規作成
torch.multiprocessing を使用し、8つの異なるモデルを独立したプロセスで同時に学習するスクリプトを作成しました。

モデル一括学習: modelA から modelH の 8 つのプロセスを同時起動します。
リソース分散: 利用可能な GPU 数に応じて各プロセスを自動的に振り分けます。
バッチサイズ指定: 256, 1024, 4096 などのバッチサイズを引数 --batch で指定可能です。
独立ログ管理: 各モデルの進捗ログ（.log）と損失グラフ（.png）を個別に保存します。
堅牢性: matplotlib がインストールされていない環境でも動作するように、グラフ描画を任意（Optional）にしました。
2. model/model.py のインポート修正
プロジェクト内のファイル名（brock）とコード内のインポート文（block）が不一致だったため、インポート文を実際のファイル名に合わせて修正しました。これにより、ModuleNotFoundError が解消されました。

実行方法
以下のコマンドで 8 つのモデルの並列学習を開始できます：

powershell
.\venv\Scripts\python.exe train\parallelization_train.py ^
    --train-data data/train.hcpe ^
    --test-data data/test.hcpe ^
    --batch 1024 ^
    --epochs 10
検証結果
インポート確認: model.model からの create_model が正常にロードできることを確認。
ヘルプ出力: --help 引数で各オプションが正しく表示されることを確認。
TIP

GPU メモリが非常に多い環境でも、8つのプロセスが同時にデータをロードするため、CPU RAM（システムメモリ）の消費量に注意してください。
```