import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. CNNの基本ブロック (Residual Block)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# 2. R-GCNブロック (テンソル情報を扱うGNN)
# 将棋の「効き」の種類（リレーション）ごとに情報を集約します
class RGCNBlock(nn.Module):
    def __init__(self, channels, num_relations):
        super(RGCNBlock, self).__init__()
        self.channels = channels
        self.num_relations = num_relations
        # 各リレーション（駒の動き）ごとの重み
        self.weight = nn.Parameter(torch.Tensor(num_relations, channels, channels))
        nn.init.xavier_uniform_(self.weight)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, adj):
        """
        x: (batch, channels, 9, 9) のテンソル
        adj: (batch, num_relations, 81, 81) の隣接行列（駒の効き）
        """
        batch_size = x.size(0)
        # (B, C, 9, 9) -> (B, 81, C) に変換
        h = x.view(batch_size, self.channels, 81).permute(0, 2, 1)

        # R-GCNのメッセージパッシング計算
        # 各リレーションごとに隣接行列を掛け、重みを適用する
        out_h = torch.zeros_like(h)
        for r in range(self.num_relations):
            # adj[:, r] は (B, 81, 81)
            # h は (B, 81, C)
            msg = torch.matmul(adj[:, r], h) # 周囲の情報を集約
            out_h += torch.matmul(msg, self.weight[r]) # 種類別の重みを適用

        # (B, 81, C) -> (B, C, 9, 9) に戻す
        out = out_h.permute(0, 2, 1).view(batch_size, self.channels, 9, 9)
        return F.relu(self.bn(out + x)) # 残差接続

# 3. ハイブリッド評価関数メインモデル
class HybridAlphaZeroNet(nn.Module):
    def __init__(self, input_channels, num_actions, num_relations=8):
        super(HybridAlphaZeroNet, self).__init__()
        self.num_relations = num_relations

        # 入力層 (例: 119チャンネル -> 256チャンネル)
        self.conv_input = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # 共通バックボーン (例: 20ブロック)
        # ここをリストの組み換えで、5つの実験パターンに変更
        self.blocks = nn.ModuleList()

        # 例: パターンC (CNN2 -> GNN2 を繰り返す)
        for _ in range(5):
            self.blocks.append(ResBlock(256))
            self.blocks.append(ResBlock(256))
            self.blocks.append(RGCNBlock(256, num_relations))
            self.blocks.append(RGCNBlock(256, num_relations))

        # Policy Head (指し手予測)
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, num_actions)
        )

        # Value Head (勝率予測)
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

    def forward(self, x, adj):
        h = self.conv_input(x)

        # 各ブロックを順番に適用
        for block in self.blocks:
            if isinstance(block, RGCNBlock):
                h = block(h, adj)
            else:
                h = block(h)

        policy = self.policy_head(h)
        value = self.value_head(h)

        return F.log_softmax(policy, dim=1), value

# --- 使用例 ---
if __name__ == "__main__":
    # パラメータ設定 (将棋を想定)
    input_ch = 119    # 入力ビットプレーン数
    actions = 2187    # 指し手の種類数 (概算)
    relations = 8     # 駒の動きの種類 (歩, 香, 桂, 銀, 金, 角, 飛, 王)

    model = HybridAlphaZeroNet(input_ch, actions, relations)

    # ダミーデータ作成
    dummy_input = torch.randn(1, input_ch, 9, 9)
    dummy_adj = torch.ones(1, relations, 81, 81) # 本来は駒の効きに基づいて作成する

    # 予測
    p, v = model(dummy_input, dummy_adj)
    print(f"Policy Output Shape: {p.shape}") # (1, 2187)
    print(f"Value Output: {v.item()}")       # -1 ~ 1 の勝率