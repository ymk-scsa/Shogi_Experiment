"""
train.py
棋譜データ（HCPEフォーマット）を用いた教師あり学習スクリプト。
shogiAI の train.py を GNN_Experiment_20251229 の HybridAlphaZeroNet 用に移植。

損失関数:
  - policy: NLLLoss（HybridAlphaZeroNet の出力が log_softmax 済みのため）
  - value:  MSELoss（Tanh 出力 [-1, 1] を [0, 1] に変換してから計算）
"""

"""具体的な実行手順
# Shogi_Experience フォルダに移動してから実行
cd c:\Users\tkksn\Desktop\Shogi_Experience\Shogi_Experience\GNN_Experiment_20251229

python train/train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:1 ^
  --batch 1024 ^
  --gpu 0
"""

"""1つのモデルを学習するコマンド
python train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:1 ^
  --batch 1024 ^
  --gpu 0 ^
  --log train_modelA.log
"""
"""複数のモデルを同時に学習するコマンド
python train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:5,modelB:10 ^
  --batch 512 ^
  --gpu 0
"""
"""中断から再開するコマンド
python train.py ^
  --train-data data/train.hcpe ^
  --test-data data/test.hcpe ^
  --models modelA:5 ^
  --resume checkpoint-modelA-step2500-interrupted.pth
"""
#python train.py ... --resume checkpoint-modelA.pth