import torch
import torch.nn as nn
import torch.nn.functional as F
# model_block.pyからすべてのブロックをインポート
from .cnn_block import ResBlock, DenseBlock, ResNeXtBlock, ConvNeXtBlock, XceptionBlock, InceptionBlock, InceptionV3Block, InceptionV4Block, InceptionResNetBlock
from .gnn_block import DynamicGNNBlock, DynamicGraphBlock, GCNBlock, GraphSAGEBlock, GINBlock, GCNIIBlock, SGCBlock, GATv2Block
from .transformer_block import RTGNNBlock, GATBlock, ViTBlock, SwinBlock, DeiTBlock, BEiTBlock, MAEBlock, DINOBlock, LocalAttentionBlock
from .others_block import SetBlock, SlotBlock, MobileNetV1Block, MobileNetV2Block, MobileNetV3Block, ShuffleNetV1Block, ShuffleNetV2Block, SqueezeNetBlock, MNasNetBlock, EfficientNetBlock, CapsuleBlock, MLPMixerBlock, SqueezeExcitationBlock, UNetBlock

class HybridAlphaZeroNet(nn.Module):
    def __init__(self, input_channels, num_actions, blocks_config):
        """
        blocks_config: ['CRE', 'GDS', 'CNX', 'GDG', ...] といったリスト形式の構成
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
            'GSC': lambda: SGCBlock(256),
            'GA2': lambda: GATv2Block(256),
            'TRT': lambda: RTGNNBlock(256),
            'TGA': lambda: GATBlock(256),
            'TVT': lambda: ViTBlock(256),
            'TSW': lambda: SwinBlock(256),
            'TDE': lambda: DeiTBlock(256),
            'TBT': lambda: BEiTBlock(256),
            'TMA': lambda: MAEBlock(256),
            'TDN': lambda: DINOBlock(256),
            'TLA': lambda: LocalAttentionBlock(256),
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
        self.backbone = nn.ModuleList()
        gcnii_count = 0
        for b in blocks_config:
            if b == 'G2I':
                gcnii_count += 1
                self.backbone.append(GCNIIBlock(256, layer_idx=gcnii_count))
            else:
                self.backbone.append(self.block_map[b]())

        # Policy Head: 次の指し手の確率
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1), #カーネルサイズの圧縮を2から32に変更
            nn.BatchNorm2d(2),
            nn.GELU(), #活性化関数をReLUから勾配損失に強いGELUに変更
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, num_actions)
        )

        # Value Head: 勝率評価 (-1 〜 1) 
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1), # ここも1から16に少しチャンネルを増やす
            nn.BatchNorm2d(16),
            nn.GELU(), #活性化関数をReLUから勾配損失に強いGELUに変更
            nn.Flatten(),
            nn.Linear(16 * 9 * 9, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh() # 勝率 -1 〜 1
        )
        
        
        # --- 補助タスクヘッド ---
        # 1. King Safety (自玉・敵玉の安全度: Scalar 2)
        self.head_king_safety = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 2)
        )
        # 2. Material (駒得: Scalar 1)
        self.head_material = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1)
        )
        # 3. Mobility (指し手の選択肢の多さ: Scalar 1)
        self.head_mobility = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1)
        )
        
        # Attack Map
        self.head_attack = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Threat Map
        self.head_threat = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Damage Map
        self.head_damage = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # -----------------------------
        # Self-Supervised heads
        # -----------------------------
        self.num_piece_types = 31 # 盤上の駒の種類数（例: 歩、香、桂、銀、金、角、飛、王など）×先手・後手と空マス
        
        self.head_masked_board = nn.Conv2d(
            256,
            self.num_piece_types,
            kernel_size=1
        )
        
        self.head_attack_map_ssl = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.head_move_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )
        
        self.head_future = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256)
        )
        
        self.head_search = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256)
        )
        

    def forward(self, x, return_aux=True):
        h = self.conv_input(x)

        for block in self.backbone:
            h = block(h)
        
        policy = self.policy_head(h)
        value = self.value_head(h)

        # return_aux=False は train/eval を問わず優先する。
        # 学習時は生logits（CrossEntropyLoss前提）、推論時はlog_softmaxを返す。
        if not return_aux:
            if self.training:
                return policy, value
            return F.log_softmax(policy, dim=1), value

        # 補助タスクの計算
        aux = {
            'king_safety': self.head_king_safety(h),
            'material': self.head_material(h),
            'mobility': self.head_mobility(h),

            'attack': self.head_attack(h),
            'threat': self.head_threat(h),
            'damage': self.head_damage(h),

            'masked_board': self.head_masked_board(h),
            'attack_map_ssl': self.head_attack_map_ssl(h),
            'move_feature': self.head_move_feature(h),
            'future_rep': self.head_future(h),
            'search_rep': self.head_search(h),
            
            'policy_entropy': self.head_policy_entropy(h),
        }
        return policy, value, aux
# ---------------------------------------------------------
# 実験設定の定義用ヘルパー
# ---------------------------------------------------------
def create_model(input_channels, num_actions, mode="30blocks"):
    """
    mode に応じて異なる構造のモデルを生成する。
    """
    if mode == "modelA":
        # モデルA: CNN 10ブロック
        config = ['CNX'] * 15 + ['CNT'] * 15

    elif mode == "modelB":
        # モデルB: CNN 5ブロック + GCN 5ブロック (計10)
        config = ['CRE'] * 20 + ['OSE'] * 10

    elif mode == "modelC":
        # モデルC: CNN 30ブロック
        config = ['CRE'] * 5 + ['CDN'] * 5 + ['CNX'] * 5 + ['CNT'] * 5 + ['CGL'] * 5 + ['CI3'] * 5

    elif mode == "modelD" or mode == "30blocks":
        # モデルD: (CNN 5 + GCN 5) を 3回繰り返す (計30)
        config = ['OM3'] * 15 + ['OSB'] * 15
    
    elif mode == "modelE" or mode == "30blocks":
        # モデルE: (CNN 5 + GCN 5) を 3回繰り返す (計30)
        config = ['CI4'] * 30
        
    elif mode == "modelF" or mode == "30blocks":
        # モデルF: (CNN 5 + GCN 5) を 3回繰り返す (計30)
        config = (['CRE'] * 5 + ['GSC'] * 5) * 3
        
    elif mode == "modelG" or mode == "30blocks":
        # モデルG: (CNN 5 + GCN 5) を 3回繰り返す (計30)
        config = ['CRE'] * 10 + ['GSC'] * 10 + ['GIN'] * 10
        
    elif mode == "modelH" or mode == "30blocks":
        # モデルH: (CNN 5 + GCN 5) を 3回繰り返す (計30)
        config = (['CRE'] * 3 + ['GSC'] * 2 + ['TLA'] * 1) * 5
        
    elif mode == "fastA":
        # モデルB: OMM 5ブロック + OSE 5ブロック (計10)
        config = ['OMM'] * 5 + ['OSE'] * 5

    else:
        # デフォルト設定（どれにも当てはまらない場合）
        config = ['CRE'] * 10
        print(f"Unknown mode: {mode}. Using default CNN 10 blocks.")
        
    return HybridAlphaZeroNet(input_channels, num_actions, config)