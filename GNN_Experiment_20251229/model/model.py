import torch
import torch.nn as nn
import torch.nn.functional as F
# model_block.pyからすべてのブロックをインポート
from .gnn_block import ResBlock
from .gnn_block import ResBlock, DynamicGNNBlock, RTGNNBlock, DynamicGraphBlock, GATBlock, SetBlock, SlotBlock, GCNBlock
from .transformer_block import RTGNNBlock
from .others_block import SetBlock

class HybridAlphaZeroNet(nn.Module):
    def __init__(self, input_channels, num_actions, blocks_config):
        """
        blocks_config: ['C', 'G', 'C', 'G', ...] といったリスト形式の構成
        """
        super(HybridAlphaZeroNet, self).__init__()

        # 入力層: 盤面情報を256チャンネルに拡張
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # ブロック生成用のマップ
        self.block_map = {
            'C': lambda: ResBlock(256),
            'G': lambda: DynamicGNNBlock(256),
            'T': lambda: RTGNNBlock(256),
            'D': lambda: DynamicGraphBlock(256),
            'A': lambda: GATBlock(256),
            'S': lambda: SetBlock(256),
            'O': lambda: SlotBlock(256),
            'N': lambda: GCNBlock(256),
        }

        # 設定リストに基づいてバックボーン（背骨）を構築
        self.backbone = nn.ModuleList([self.block_map[b]() for b in blocks_config])

        # Policy Head: 次の指し手の確率
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, num_actions)
        )

        # Value Head: 勝率評価 (-1 〜 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.conv_input(x)
        for block in self.backbone:
            h = block(h)
        
        policy = self.policy_head(h)
        value = self.value_head(h)
        return F.log_softmax(policy, dim=1), value

# ---------------------------------------------------------
# 実験設定の定義用ヘルパー
# ---------------------------------------------------------
def create_model(input_channels, num_actions, mode="30blocks"):
    """
    mode に応じて異なる構造のモデルを生成する。
    """
    if mode == "modelA":
        # モデルA: CNN 10ブロック
        config = ['C'] * 10

    elif mode == "modelB":
        # モデルB: CNN 5ブロック + GCN 5ブロック (計10)
        config = ['C'] * 5 + ['N'] * 5

    elif mode == "modelC":
        # モデルC: CNN 30ブロック
        config = ['C'] * 30

    elif mode == "modelD" or mode == "30blocks":
        # モデルD: (CNN 5 + GCN 5) を 3回繰り返す (計30)
        config = (['C'] * 5 + ['N'] * 5) * 3

    else:
        # デフォルト設定（どれにも当てはまらない場合）
        config = ['C'] * 10
        print(f"Unknown mode: {mode}. Using default CNN 10 blocks.")
        
    return HybridAlphaZeroNet(input_channels, num_actions, config)