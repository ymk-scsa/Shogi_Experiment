```
GNN_Experience_20251229/
├── model.py          # 先ほど作成したハイブリッドモデル
├── game/
│   └── board.py      # 将棋のルール、駒の動き、合法手生成
├── search/
│   └── search.py     # モンテカルロ木探索の実装
├── agent/
│   └── player.py     # MCTSとModelを組み合わせて指し手を決める
├── data/
│   └── buffer.py     # 自己対局の棋譜を保存するリプレイバッファ
├── train.py          # 学習ループ（自己対局 → 学習 → モデル更新）
├── weights/          # 学習済みモデルの保存先
├── reqirements.txt         # 環境構築(venv)など
└── README.md          # メモ
```

```
1. CNNブロック (Residual Block) C
2. 自動グラフ構築型GNNブロック (Vision GNN Style) G 初期版　計算量重
3 'T' = RT-GNN（Relational Token GNN）ブロック
4 'D' = Dynamic Graph GNN（EdgeConv系・shallow）ブロック
5 'A' = GAT（Graph Attention, 駒ではなく81トークン版）ブロック (GAT = attention + adjacency)6 'S' = 6 6 'S' = Set Transformer / Deep Sets ブロック# 元の入力(r)を足し（スキップ接続）
7 'O' = Object-centric / Slot Attention（軽量版）ブロック
8 'N' = GCN（固定近傍グラフ）ブロック(DeepSets / Global Context Injection)

モデル実験構築例
def create_model(input_channels, num_actions, mode="30blocks"):
    """
    mode に応じて異なる構造のモデルを生成する。
    'modelA' 〜 'modelH' のラベルで最大8個のバリエーションを定義可能。
    """
    if mode == "modelA" or mode == "30blocks":
        # CNN 5 + GNN 5 を 3回 (計30)
        config = (['C'] * 5 + ['G'] * 5) * 3
    elif mode == "modelB" or mode == "40blocks":
        # CNN 5 + GNN 5 を 4回 (計40)
        config = (['C'] * 5 + ['G'] * 5) * 4
    elif mode == "modelC":
        # 深いGNN重視
        config = (['G'] * 10) * 2
    elif mode == "modelD":
        # 深いCNN重視
        config = (['C'] * 10) * 2
    elif mode == "modelE":
        # RTGNN主体
        config = (['T'] * 5) * 4
    elif mode == "modelF":
        # GAT主体
        config = (['A'] * 5) * 4
    elif mode == "modelG":
        # ハイブリッド構成 A
        config = ['C', 'G', 'T', 'D', 'A', 'S', 'O', 'N'] * 3
    elif mode == "modelH":
        # ハイブリッド構成 B
        config = (['C', 'G', 'T', 'N'] * 5)
    else:
        # デフォルト (modelA相当)
        config = (['C'] * 5 + ['G'] * 5) * 3

PCに強力なグラフィックボード（NVIDIA製）が載っていない場合は、CPUモード（-g -1） で動かす

# 1. デスクトップなど、作業したい場所に移動
cd %USERPROFILE%\Desktop

# 2. プロジェクトフォルダに移動（すでにフォルダがある場合）
cd Shogi_Experience\Shogi_Experience\GNN_Experiment_20251229

# 3. 仮想環境（venv）を作成
# これを作ることで、PC本体の環境を汚さずに済みます
python -m venv venv

# 4. 仮想環境を起動（有効化）
# 左側に (venv) と表示されれば成功です
.\venv\Scripts\activate

# ライブラリのインストール
pip install -r requirements.txt

# もし PyTorch が入らなければ、手動でインストール
pip install torch torchvision torchaudio

GPU（NVIDIA製）を積んでいるPCの場合
python train.py --train-data data/Suisho10Mn_psv.bin --test-data data/Suisho10Mn_psv.bin --models modelD:1 --format psv --gpu 0

普通のPC（CPU）で動かす場合
python train.py --train-data data/Suisho10Mn_psv.bin --test-data data/Suisho10Mn_psv.bin --models modelD:1 --format psv --gpu -1
```