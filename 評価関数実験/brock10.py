import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, ClusterGCNConv

class ViGBlock(nn.Module):
    """GNNを用いた関係性抽出ブロック"""
    def __init__(self, channels):
        super().__init__()
        self.gnn = GraphConv(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch_size):
        # x: [Batch * 81, Channels]
        residual = x
        out = self.gnn(x, edge_index)
        out = self.norm(out)
        out = self.relu(out)
        return out + residual

class CNNBlock(nn.Module):
    """CNNを用いた局所パターン抽出ブロック"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [Batch, Channels, 9, 9]
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out + residual

class HybridBlock(nn.Module):
    """CNNとViGを組み合わせた1単位（合計2層分）"""
    def __init__(self, channels):
        super().__init__()
        self.cnn_layer = CNNBlock(channels)
        self.vig_layer = ViGBlock(channels)

    def forward(self, x, edge_index):
        # 1. CNN層 (空間ドメイン)
        x = self.cnn_layer(x)
        
        # 2. ViG層 (グラフドメイン) への変換
        B, C, H, W = x.shape
        x_graph = x.permute(0, 2, 3, 1).reshape(-1, C) # [B*81, C]
        
        # 3. ViG層
        x_graph = self.vig_layer(x_graph, edge_index, B)
        
        # 4. 画像形式へ戻す
        x = x_graph.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class ShogiDeepHybridNet(nn.Module):
    def __init__(self, input_channels=119, num_blocks=10, hidden_channels=256):
        super().__init__()
        # 初期畳み込み (Stem)
        self.stem = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        
        # 10個のHybridBlock (計20層の隠れ層)
        self.layers = nn.ModuleList([
            HybridBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Policy Head: 指し手の予測 (1393は将棋の全合法手ラベル数の例)
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 1393)
        )
        
        # Value Head: 勝率の予測 (-1.0 ~ 1.0)
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x, edge_index):
        # x: [Batch, Channels, 9, 9]
        # edge_index: [2, Total_Edges_in_Batch]
        
        x = F.relu(self.stem(x))
        
        for layer in self.layers:
            x = layer(x, edge_index)
            
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value