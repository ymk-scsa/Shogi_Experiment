import torch
import torch.nn as nn
import torch.nn.functional as F
# model_block.pyからすべてのブロックをインポート
from .cnn_block import ResBlock, DenseBlock, ResNeXtBlock, ConvNeXtBlock, XceptionBlock, InceptionBlock, InceptionV3Block, InceptionV4Block, InceptionResNetBlock
from .gnn_block import DynamicGNNBlock, DynamicGraphBlock, GCNBlock, GraphSAGEBlock, GINBlock, GCNIIBlock, SGCBlock, GATv2Block
from .transformer_block import RTGNNBlock, GATBlock, ViTBlock, SwinBlock, DeiTBlock, BEiTBlock, MAEBlock, DINOBlock
from .others_block import SetBlock, SlotBlock, MobileNetV1Block, MobileNetV2Block, MobileNetV3Block, ShuffleNetV1Block, ShuffleNetV2Block, SqueezeNetBlock, MNasNetBlock, EfficientNetBlock, CapsuleBlock, MLPMixerBlock, SqueezeExcitationBlock, UNetBlock

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
            'CRE': lambda: ResBlock(256),
            'CDN': lambda: DenseBlock(256),
            'CNX': lambda: ResNeXtBlock(256),
            'CNT': lambda: ConvNeXtBlock(256),
            'CXC': lambda: XceptionBlock(256),
            'CGL': lambda: InceptionBlock(256),
            'CI3': lambda: InceptionV3Block(256),
            'CI4': lambda: InceptionV4Block(256),
            'CIR': lambda: InceptionResNetBlock(256),
            'GDS': lambda: DynamicGNNBlock(256),
            'GDG': lambda: DynamicGraphBlock(256),
            'GCN': lambda: GCNBlock(256),
            'GSG': lambda: GraphSAGEBlock(256),
            'GIN': lambda: GINBlock(256),
            'G2I': lambda: GCNIIBlock(256),
            'GSG': lambda: SGCBlock(256),
            'GA2': lambda: GATv2Block(256),
            'TRT': lambda: RTGNNBlock(256),
            'TGA': lambda: GATBlock(256),
            'TVT': lambda: ViTBlock(256),
            'TSW': lambda: SwinBlock(256),
            'TDE': lambda: DeiTBlock(256),
            'TBT': lambda: BEiTBlock(256),
            'TMA': lambda: MAEBlock(256),
            'TDN': lambda: DINOBlock(256),
            'OST': lambda: SetBlock(256),
            'OSB': lambda: SlotBlock(256),
            'OM1': lambda: MobileNetV1Block(256),
            'OM2': lambda: MobileNetV2Block(256),
            'OM3': lambda: MobileNetV3Block(256),
            'OS1': lambda: ShuffleNetV1Block(256),
            'OS2': lambda: ShuffleNetV2Block(256),
            'OSQ': lambda: SqueezeNetBlock(256),
            'OMN': lambda: MNasNetBlock(256),
            'OEF': lambda: EfficientNetBlock(256),
            'OCB': lambda: CapsuleBlock(256),
            'OMM': lambda: MLPMixerBlock(256),
            'OSE': lambda: SqueezeExcitationBlock(256),
            'OUN': lambda: UNetBlock(256),
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