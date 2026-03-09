import torch
import torch.nn as nn
import torch.nn.functional as F

#3 'T' = RT-GNN（Relational Token GNN）ブロック
# トークン間の関係性を捉える自己注意機構を用いたGNNブロック
class RTGNNBlock(nn.Module): # Transformerの仕組みをグラフネットワークに応用したRTGNNブロックの定義
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels) # レイヤー正規化：各マスのデータの平均と分散を整え、学習の安定性を高める
        self.attn = nn.MultiheadAttention( # マルチヘッド・アテンション：全81マスが「どのマスに注目すべきか」を、4つの異なる視点(heads)で計算する
            embed_dim=channels, # 入力される情報の深さ
            num_heads=heads, # 注目ポイントを並列で探す数
            batch_first=True # データの並びを (バッチ, マス, 情報) の順にする設定
        )
        self.ffn = nn.Sequential( # フィードフォワード・ネットワーク：アテンションで得られた情報をさらに洗練させるための小さな層
            nn.Linear(channels, channels), # 全結合層：情報を線形変換する
            nn.GELU() # GELU活性化関数：滑らかな変化を加え、高度な特徴を抽出する
        )

    def forward(self, x):
        B, C, H, W = x.shape # 入力xの形状（バッチ, チャンネル, 高さ9, 幅9）を取得
        r = x # あとで足し戻すために、元の入力データを保存（残差接続）

        tokens = x.view(B, C, -1).permute(0, 2, 1)  # (B,81,C) 形状: (バッチ, 81マス, 情報C)
        tokens = self.norm(tokens) # 各トークン（マス）を正規化する

        attn_out, _ = self.attn(tokens, tokens, tokens) # 全マスの相互注目（Self-Attention）を実行 # tokensを3つ渡すのは、自分自身を「検索条件(Q)」「対象(K)」「内容(V)」のすべてに使うため
        tokens = tokens + self.ffn(attn_out) # アテンションの結果にフィードフォワードの変換を加え、元のトークンに足し合わせる

        out = tokens.permute(0, 2, 1).view(B, C, H, W) # 1列に並んでいたトークンを、元の (B, C, 9, 9) の盤面形状に戻す
        return F.relu(out + r) # 元の入力(r)を足し、最後にReLUで活性化して結果を返す
    
#5 'A' = GAT（Graph Attention, 駒ではなく81トークン版）ブロック (GAT = attention + adjacency)
# グラフ注意機構を用いて、各ノードが重要な隣接ノードから情報を集約するGNNブロック
class GATBlock(nn.Module): # グラフ・アテンション・ネットワーク(GAT)の仕組みを応用したブロック
    def __init__(self, channels, heads=4):
        super().__init__() # マルチヘッド・アテンション：全マスの関係性を「4つの異なる視点(heads)」で同時に分析する
        self.attn = nn.MultiheadAttention( # 各マスが「どのマスに注目すべきか」の重みを学習によって決定する
            embed_dim=channels, # 1マスの情報量（チャンネル数）
            num_heads=heads, # 注目ポイントを並列で探す数（多いほど多角的に分析できる）
            batch_first=True # データの並びを (バッチ, マス, 情報) の順で扱う設定
        )
        self.norm = nn.LayerNorm(channels) # レイヤー正規化：データの数値を安定させ、アテンションの計算がうまくいくように整える

    def forward(self, x):
        B, C, H, W = x.shape # 入力の形状（バッチサイズ, チャンネル, 高さ9, 幅9）を取得
        r = x # 後で足し合わせる（残差接続）ために、元のデータを保存

        tokens = x.view(B, C, -1).permute(0,2,1) # 9x9の盤面を「81個のトークン（情報の塊）」として1列に並べ替える (B, C, 81) にしてから (B, 81, C) に軸を変換
        tokens = self.norm(tokens) # 並べたトークンに対して正規化を適用

        # 自己注意（Self-Attention）メカニズムを実行
        # tokensを3つ（Q, K, V）として渡すことで、81マスの全組み合わせの相性を計算する
        out, _ = self.attn(tokens, tokens, tokens) # outには、他のマスの情報を「重要度（Attention Weight）」に応じて混ぜ合わせた結果が入る
        out = out.permute(0,2,1).view(B,C,H,W) # 処理のために並べ替えていた軸を戻し、(B, C, 9, 9) の盤面形状に復元する

        return F.relu(out + r) # 元の入力(r)を足し（スキップ接続）、ReLUで活性化して次の層へ渡す