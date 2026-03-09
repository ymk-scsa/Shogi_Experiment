import torch
import torch.nn as nn
import torch.nn.functional as F

#6 'S' = Set Transformer / Deep Sets ブロック# 元の入力(r)を足し（スキップ接続）、ReLUで活性化して次の層へ渡す
# 集合データを扱うためのTransformerベースのブロック
class SetBlock(nn.Module): # 盤面全体の「集合的（Set）」な情報を抽出し、各マスに共有するブロック
    def __init__(self, channels):
        super().__init__()
        self.phi = nn.Sequential( # 各マスの情報を「全体としての特徴」に変換するためのネットワーク
            nn.Linear(channels, channels),
            nn.ReLU()
        )
        self.rho = nn.Sequential( # 全体の特徴（平均）を、再び各マスの情報へ統合するためのネットワーク
            nn.Linear(channels, channels),
            nn.ReLU()
        )

    def forward(self, x): 
        B, C, H, W = x.shape # 入力の形状（バッチ, チャンネル, 高さ9, 幅9）を取得
        r = x # あとで足し合わせる（残差接続）ために元のデータを保存

        tokens = x.view(B, C, -1).permute(0,2,1)  # (B,81,C) 盤面を81個の「要素（トークン）」として1列に展開
        # 1. 盤面全体の情報を集約
        # 各マスの情報をphiで加工してから、全81マスの「平均（mean）」を計算する
        # これにより、盤面全体の「厚み」や「戦況の要約」が1つのベクトルに凝縮される
        pooled = self.phi(tokens).mean(dim=1, keepdim=True) # 形状: (B, 1, C)
        # 2. 全体情報を各マスに配分
        # 得られた全体の要約（pooled）をrhoで加工し、元の各マス（tokens）に足し合わせる
        # これにより、各マスの駒の情報に「盤面全体の状況」という追加情報が加わる
        tokens = tokens + self.rho(pooled) 

        out = tokens.permute(0,2,1).view(B,C,H,W) # 処理のために並べ替えていた軸を戻し、(B, C, 9, 9) の形状に復元
        return F.relu(out + r) # 元の入力(r)を足し、ReLUで活性化して出力

#7 'O' = Object-centric / Slot Attention（軽量版）ブロック
# オブジェクト中心の注意機構を用いて、重要な特徴を抽出するブロック
class SlotBlock(nn.Module): # 特定の「役割（スロット）」に情報を集約して盤面を解析するブロック
    def __init__(self, channels, slots=16): # 16個の「情報の受け皿（スロット）」を学習可能なパラメータとして定義
        super().__init__()
        self.slots = nn.Parameter( # 初期値は小さな乱数(0.02倍)で設定し、学習を通じて「注目すべきパターン」を覚える
            torch.randn(slots, channels)*0.02 #係数は修正の余地あり
        )
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True) # スロットが盤面のどのマスに注目するかを計算するためのアテンション機構（4ヘッド）

    def forward(self, x):
        B, C, H, W = x.shape # 入力の形状（バッチ, チャンネル, 高さ9, 幅9）を取得
        r = x # あとで足し合わせる（残差接続）ために元のデータを保存

        # 盤面を81個の「トークン（マスの情報）」に展開
        tokens = x.view(B, C, -1).permute(0,2,1) # 形状: (B, 81, C)
        slots = self.slots.unsqueeze(0).expand(B, -1, -1) # 形状: (B, 16, C)

        # アテンション実行：スロット(Q)が盤面の各マス(K, V)から情報を吸い上げる
        # これにより、16個のスロットそれぞれが「王様の周り」「攻めの拠点」などの特徴を掴む
        slot_out, _ = self.attn(slots, tokens, tokens)
        pooled = slot_out.mean(dim=1, keepdim=True) # 16個のスロットが得た情報を平均し、盤面全体の「要約」を1つ作る 形状: (B, 1, C)

        tokens = tokens + pooled # 抽出された重要な要約情報を、元の81マスの情報に足し合わせる
        out = tokens.permute(0,2,1).view(B,C,H,W) # 1列に並んだデータを元の (B, C, 9, 9) の盤面形状に戻す

        return F.relu(out + r) # 元の入力(r)を足し、ReLUで活性化して出力