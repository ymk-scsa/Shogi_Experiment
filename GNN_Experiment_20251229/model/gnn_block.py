import torch
import torch.nn as nn
import torch.nn.functional as F

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