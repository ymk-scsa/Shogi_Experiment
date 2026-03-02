import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class HybridLayer(nn.Module):
    def __init__(self, channels):
        super(HybridLayer, self).__init__()
        # CNN Branch: 局所的なパターンを捉える
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # ViG Branch: 広域的な関係（駒の利きなど）を捉える
        self.gnn = GraphConv(channels, channels)
        self.gnn_bn = nn.BatchNorm1d(channels)

    def forward(self, x, edge_index):
        # xの形状: [Batch, Channels, 9, 9]
        residual = x
        
        # 1. CNN層の処理
        out = self.cnn(x)
        
        # 2. ViG層の処理（画像をグラフ形式に変換）
        B, C, H, W = out.shape
        # [B, C, H, W] -> [B*H*W, C] に変換してGNNへ
        out_gnn = out.permute(0, 2, 3, 1).reshape(-1, C)
        
        # グラフ畳み込み（edge_indexは事前に定義した駒の利きなど）
        out_gnn = self.gnn(out_gnn, edge_index)
        out_gnn = self.gnn_bn(out_gnn)
        out_gnn = F.relu(out_gnn)
        
        # 3. 再び画像の形状 [Batch, Channels, 9, 9] に戻す
        out = out_gnn.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # スキップ接続
        return out + residual

class ShogiAlphaViG(nn.Module):
    def __init__(self, input_channels, num_blocks, hidden_channels):
        super(ShogiAlphaViG, self).__init__()
        # 初手の特徴抽出 (Stem)
        self.stem = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        
        # Hybrid層（CNNとViGの交互ブロック）を積み重ねる
        self.layers = nn.ModuleList([
            HybridLayer(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Policy Head (指し手予測)
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 1393) # 将棋の指し手総数などの例
        )
        
        # Value Head (勝率予測)
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(1 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x, edge_index):
        x = F.relu(self.stem(x))
        
        for layer in self.layers:
            x = layer(x, edge_index)
            
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

# ダミーデータでの実行例
# 入力: [バッチサイズ, チャンネル数, 9, 9]
# edge_index: [2, エッジ数] (盤面の接続関係)