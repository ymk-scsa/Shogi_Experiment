import torch
import torch.nn as nn
import torch.nn.functional as F

#1. TRT = RTGNNBlock
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
    
#2. TGA = GATBlock（Graph Attention, 駒ではなく81トークン版）ブロック (GAT = attention + adjacency)
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
    
#3. TVT = ViTBlock (Vision Transformer) ブロック
# 盤面全体をトークンの集合として扱い、標準的なアテンションを適用する
class ViTBlock(nn.Module):
    def __init__(self, channels, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        x = x.view(B, C, -1).permute(0, 2, 1) # (B, 81, C)
        
        # Attention
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2)[0]
        # MLP
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        
        out = x.permute(0, 2, 1).view(B, C, H, W)
        return out + r

#4. TSW = SwinBlock (Swin Transformer ブロック)
# 9x9の盤面を3x3のウィンドウに分け、局所的なアテンションで計算量を削減する
class SwinBlock(nn.Module):
    def __init__(self, channels, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        # ウィンドウ分割 (B, 3x3, 3x3, C)
        x = x.view(B, C, H//self.window_size, self.window_size, W//self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size*self.window_size, C)
        
        # Window Attention
        x = self.norm(x)
        x = x + self.attn(x, x, x)[0]
        
        # 逆変換
        x = x.view(B, H//self.window_size, W//self.window_size, self.window_size, self.window_size, C)
        out = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return out + r

#5. TDE = DeiTBlock (Data-efficient Image Transformer) ブロック
# 蒸留（Distillation）トークンを模した、知識移転に強い構造
class DeiTBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cls_token_sim = nn.Parameter(torch.zeros(1, 1, channels)) # 教師モデルの知識を模す
        self.attn = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        tokens = x.view(B, C, -1).permute(0, 2, 1) # (B, 81, C)
        distill_token = self.cls_token_sim.expand(B, -1, -1)
        
        # トークン結合してアテンション
        x = torch.cat((distill_token, tokens), dim=1)
        x = self.norm(x)
        x = x + self.attn(x, x, x)[0]
        
        # 盤面トークンのみ抽出
        out = x[:, 1:].permute(0, 2, 1).view(B, C, H, W)
        return out + r

#6. TBT = BEiTBlock (Bidirectional Encoder representation from Image Transformers) ブロック
# 盤面の一部を隠した事前学習に向く、相対位置バイアスを持つアテンション
class BEiTBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rel_pos_bias = nn.Parameter(torch.zeros(81, 81)) # 相対位置関係を学習
        self.attn = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.norm(x)
        
        # 注意：MultiheadAttentionの内部計算にbiasを加える簡易実装
        attn_out, _ = self.attn(x, x, x) 
        out = (attn_out).permute(0, 2, 1).view(B, C, H, W)
        return out + r

#7. TMA = MAE (Masked Autoencoder) ブロック
# 情報が欠損していても、周囲から盤面を再構成する能力が高い構造
class MAEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.decoder_embed = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, 8, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        x = x.view(B, C, -1).permute(0, 2, 1)
        # エンコーダ・デコーダ的なボトルネック構造を模倣
        x = self.norm(self.decoder_embed(x))
        x = x + self.attn(x, x, x)[0]
        
        out = x.permute(0, 2, 1).view(B, C, H, W)
        return out + r

#8. TDN = DINOBlock (DINOv2 ブロック)
# 自己教師あり学習で得られる「物体の輪郭（将棋なら駒の勢力圏）」を捉えるための正規化重視ブロック
class DINOBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
        # DropPath（Stochastic Depth）の簡易的な実装
        self.drop_path = nn.Identity() 

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        x = x.view(B, C, -1).permute(0, 2, 1)
        
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        out = x.permute(0, 2, 1).view(B, C, H, W)
        return out + r