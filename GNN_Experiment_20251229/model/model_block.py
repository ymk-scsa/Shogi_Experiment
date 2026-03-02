"""
GNNとCNNのハイブリッドモデルのベース
グラフ定義によってGNNの制度が大幅に変化したため、
テンソル情報から自動でグラフデータを学習するVisionGNNを用いている。
計算量、推論速度に大幅に問題がある。
ここから派生モデルを複数作成

アイデア 'L' = Light Graph Convolutional Network (LightGCN) ブロック
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. CNNブロック (Residual Block)
# 盤面の局所的なパターン（3x3）を抽出する
class ResBlock(nn.Module): #ResBlockという名前の新しいネットワークのクラスを定義
    def __init__(self, channels): #初期化メソッド、channelsは入力チャネル数
        super(ResBlock, self).__init__() #PyTorchの基本クラスを初期化
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) #3x3の畳み込み層を定義。padding=1 を指定することで、画像のサイズ（盤面のサイズ）を変えずに処理。
        self.bn1 = nn.BatchNorm2d(channels) #バッチ正規化層を定義
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1) #もう一つ3x3の畳み込み層を定義
        self.bn2 = nn.BatchNorm2d(channels) #もう一つバッチ正規化層を定義

    def forward(self, x): #順伝播の定義
        residual = x #入力をresidualとして保存
        out = F.relu(self.bn1(self.conv1(x))) #1つ目の畳み込み、バッチ正規化、ReLU活性化関数を適用
        out = self.bn2(self.conv2(out)) #2つ目の畳み込みとバッチ正規化を適用
        out += residual #residual接続を追加。加工したデータ out に、最初に保存しておいた「生の入力 residual」を足し合わせる。
        return F.relu(out) #最後にReLUで仕上げて、結果を返す。

# 2. 自動グラフ構築型GNNブロック (Vision GNN Style)
# 外部からの隣接行列を必要とせず、テンソルから動的にグラフを生成する
class DynamicGNNBlock(nn.Module): #動的グラフニューラルネットワーク（Dynamic GNN）という手法を用いたネットワーク層を定義
    def __init__(self, channels, k=9): #このブロックを初期化する関数。各マスが持つ情報の深さ（ベクトルサイズ）をchannelsで指定。kは各マスが接続する近傍ノード数
        super(DynamicGNNBlock, self).__init__() #PyTorchのベースクラス（nn.Module）を初期化し、このクラスをPyTorchのモジュールとして扱えるようにする
        self.k = k  # 各マスが接続する近傍ノード数 引数で受け取った k の値を、クラス内の変数として保存

        # 特徴変換用の層
        self.fc_in = nn.Sequential( ## 入力データを変換するためのシーケンシャル（一連の処理）を定義
            nn.Conv2d(channels, channels, 1), # 1x1の畳み込み層：マスの位置関係は変えず、各マスの「情報の質」を線形変換する
            nn.BatchNorm2d(channels), ## バッチ正規化：学習を安定させるために、データの分布を平均0、分散1に近い状態に整える
            nn.GELU() # GELU活性化関数：ReLUをより滑らかにした関数で、近年の高性能なAIモデル（BERT等）でよく使われる
        )
        self.fc_out = nn.Sequential( # グラフ処理を終えた後のデータを最終調整するためのシーシャルを定義
            nn.Conv2d(channels, channels, 1), # 1x1の畳み込み層：集約された近傍ノードの情報と自分自身を混ぜ合わせ、最終的な特徴量を抽出する
            nn.BatchNorm2d(channels) # バッチ正規化：出力前のデータの偏りを取り除き、次のレイヤーへ渡しやすくする
        )

        # メッセージパッシング用の重み
        self.nn = nn.Sequential( # 2つのノード間の「関係性」を学習するための小さなニューラルネットワークを定義
            nn.Linear(channels * 2, channels), # 全結合層：自分と相手の情報を合わせた「2倍のチャンネル数」を入力し、元のサイズに圧縮する
            nn.LeakyReLU(0.2), ## LeakyReLU活性化関数：入力がマイナスのときもわずかに値を通す（0.2倍）ことで、学習が止まるのを防ぐ
            nn.Linear(channels, channels) # 全結合層：最終的な「関係性の深さ（エッジの特徴）」を出力する
        )

    def forward(self, x):
        # x: (B, C, 9, 9) 
        B, C, H, W = x.shape # 入力の形状を取得 (B:バッチサイズ, C:チャンネル, H:高さ9, W:幅9)
        residual = x # あとで足し戻すために元の入力を保存（スキップ接続用）

        # 1. 各マスをノードとして展開 (B, 81, C)
        x_in = self.fc_in(x) # 前処理用の1x1畳み込みを通し、各マスの特徴を洗練させる
        features = x_in.view(B, C, -1).permute(0, 2, 1) # (B, 81, C)

        # 2. 動的グラフ構築 (k-Nearest Neighbors)
        # 各ノード間の距離を計算
        dist = torch.cdist(features, features) # (B, 81, 81)
        # 距離が近い上位k個のインデックスを取得  _ は距離の値そのもの（今回は不要）、nn_idx はそのマスの番号（インデックス）
        _, nn_idx = torch.topk(dist, self.k, dim=-1, largest=False) # (B, 81, k)

        # 3. メッセージパッシング
        # (B, 81, k, C) の形に周辺ノードの特徴を並べる
        batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1) # バッチごとのインデックス
        node_idx = torch.arange(81, device=x.device).view(1, -1, 1) # 各マス(0-80)のインデックス
        neighbor_features = features[batch_idx, nn_idx] # (B, 81, k, C)

        # 中心ノードの特徴を拡張して結合
        center_features = features.unsqueeze(2).repeat(1, 1, self.k, 1) # (B, 81, k, C)
        edge_features = torch.cat([center_features, neighbor_features - center_features], dim=-1) # neighbor_features - center_features は自分から見た相手との違いを意味する

        # メッセージの集約 (Max Pooling)
        messages = self.nn(edge_features) # (B, 81, k, C)
        new_features, _ = torch.max(messages, dim=2) # (B, 81, C)

        # 4. 元のテンソル形状に戻す
        out = new_features.permute(0, 2, 1).view(B, C, H, W) # 処理のために並べ替えていた軸を (B, C, 81) に戻し、さらに (B, C, 9, 9) の盤面形状に復元する
        out = self.fc_out(out) # 後処理用の1x1畳み込みを通し、最終的な特徴を調整する

        return F.relu(out + residual) # 最初の入力を足し合わせ（残差接続）、ReLUで活性化して出力する
    
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
    
#4 'D' = Dynamic Graph GNN（EdgeConv系・shallow）ブロック
# 動的にグラフを構築し、浅いネットワークで効果的に特徴を抽出するGNNブロック
class DynamicGraphBlock(nn.Module): # 動的なグラフ構造を用いて盤面の特徴を抽出するブロック
    def __init__(self, channels, k=9): # 自分に関連の深い上位k個（9個）のマスを選択する設定
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential( # エッジ（マスの繋がり）の情報を計算するための多層パーセプトロン(MLP)
            nn.Linear(channels * 2, channels), # 自分と相手の情報を結合した(channels * 2)を入力し、圧縮する
            nn.LeakyReLU(0.2), # 負の値をわずかに通すことで学習の停滞を防ぐ活性化関数
            nn.Linear(channels, channels) # 最終的なエッジの特徴量を出力する
        )
        self.bn = nn.BatchNorm2d(channels) # バッチ正規化：計算結果を整えて学習効率を上げる

    def forward(self, x):
        B, C, H, W = x.shape # 入力の形状を取得 (B=バッチ, C=チャンネル, H=9, W=9)
        r = x # 残差接続（後で足すため）に元のデータを保存

        feats = x.view(B, C, -1).permute(0, 2, 1)  # (B,81,C) 盤面の9x9を81個の独立したノードに並べ替える
        dist = torch.cdist(feats, feats) # 全てのマス同士の「特徴の距離」を計算する
        _, idx = torch.topk(dist, self.k, largest=False) # 距離が近い（＝今の戦況で関係が深い）上位k個のインデックスを取得

        batch = torch.arange(B, device=x.device).view(B,1,1) # 各バッチの計算用インデックスを作成
        neighbors = feats[batch, idx]  # (B,81,k,C)
        center = feats.unsqueeze(2).expand_as(neighbors) # 自分自身の情報を、近傍ノードと同じ数だけコピーして並べる

        edge = torch.cat([center, neighbors - center], dim=-1) # 「自分」と「自分と相手の差分」を合体させて、エッジの特徴を作る
        msg = self.edge_mlp(edge).max(dim=2)[0] # エッジ情報をMLPで計算し、k個の中から最大の値（最も強い特徴）を抽出する(Max Pooling)

        out = msg.permute(0,2,1).view(B,C,H,W) #列に並んだデータを元の (B, C, 9, 9) の形状に戻す
        out = self.bn(out) # バッチ正規化を適用
        return F.relu(out + r) # 元の入力(r)を足し合わせ（スキップ接続）、ReLUで仕上げて出力

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

#8 'N' = GCN（固定近傍グラフ）ブロック(DeepSets / Global Context Injection)
# 事前定義された近傍グラフに基づいて情報を集約するGNNブロック
class GCNBlock(nn.Module): # グラフ畳み込み（GCN）の概念を簡略化した軽量な情報共有ブロック
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, channels) # 盤面全体の平均情報を、各マスに適合するように調整するための全結合層

    def forward(self, x):
        B, C, H, W = x.shape # 入力の形状（バッチ, チャンネル, 高さ9, 幅9）を取得
        r = x # 残差接続（あとで足し合わせる）ために元のデータを保存

        # 盤面の9x9を81個の「ノード（頂点）」として1列に展開
        tokens = x.view(B, C, -1).permute(0,2,1) # 形状を (B, 81, C) に変換する
        # 1. 盤面全体の要約（平均）を計算
        # 81マスすべての情報を平均し、現在の局面の「全体的な特徴」を1つのベクトルにする
        mean = tokens.mean(dim=1, keepdim=True) # 形状: (B, 1, C)
        tokens = tokens + self.linear(mean) # これにより、個々の駒が「盤面全体のムード」を考慮した状態に更新される

        out = tokens.permute(0,2,1).view(B,C,H,W) # 処理のために並べ替えていた軸を戻し、(B, C, 9, 9) の盤面形状に復元
        return F.relu(out + r) # 元の入力(r)を足し、ReLUで活性化して出力