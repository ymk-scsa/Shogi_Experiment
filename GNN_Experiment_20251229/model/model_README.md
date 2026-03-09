"""
概要
現在、DL系将棋AIにはCNNが用いられている。志柿近年では性能向上は鈍化し、表現力の限界が迫っている。

目的
CNNにGNNやTransformerなど複数のアーキテクチャを組み合わせることで、表現力向上を目指すハイブリッドアーキテクチャの研究を行う。

（例）
GNNとCNNのハイブリッドモデルのベース
グラフ定義によってGNNの制度が大幅に変化したため、
テンソル情報から自動でグラフデータを学習するVisionGNNを用いている。
計算量、推論速度に大幅に問題がある。
ここから派生モデルを複数作成

アイデア 'L' = Light Graph Convolutional Network (LightGCN) ブロック

構成
cnn_brock.py　CNN系アーキテクチャのブロック
gnn_brock.py　CNN系アーキテクチャのブロック
Transformer_brock.py　Transformer系アーキテクチャのブロック
others_brock.py　その他アーキテクチャのブロック

適宜新しいアーキテクチャの系統を追加を行う

model.py　各ブロック要素を組み合わせることで、ハイブリッドアーキテクチャを構築する
"""