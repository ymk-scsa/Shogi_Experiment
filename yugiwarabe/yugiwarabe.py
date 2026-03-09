import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Local Feature Extractor (CNN / ResNeXt)
# ============================================================
class ResNeXtBlock(nn.Module):
    """
    ResNeXt block: cardinality=32, bottleneck structure
    """
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, bottleneck_width=4):
        super().__init__()
        D = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class CNNStem(nn.Module):
    def __init__(self, input_channels, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([
            ResNeXtBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x

# ============================================================
# 2. Piece Interaction Network (GNN)
# ============================================================
class FlashGNNLayer(nn.Module):
    """
    GNN for square-to-square and square-to-hand interactions.
    We treat 81 squares + hand pieces as nodes.
    """
    def __init__(self, dim):
        super().__init__()
        self.msg_proj = nn.Linear(dim, dim)
        self.update_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x: [B, N, D], mask: [B, N, 1]
        if mask is not None:
            masked_x = x * mask
            num_nodes = mask.sum(dim=1, keepdim=True) + 1e-8
            mean_msg = masked_x.sum(dim=1, keepdim=True) / num_nodes
        else:
            mean_msg = x.mean(dim=1, keepdim=True)
            
        mean_msg = mean_msg.expand_as(x)
        h = torch.cat([x, self.msg_proj(mean_msg)], dim=-1)
        out = self.norm(x + self.update_proj(h))
        return F.relu(out)

# ============================================================
# 3. ShogiBoardTransformer (Square, Ray, Piece Attention)
# ============================================================
def get_shogi_ray_mask():
    """
    Returns a mask of shape [81, 81] where (i, j) is 0 if squares i and j
    are on the same row, column, or diagonal, and -1e9 otherwise.
    """
    mask = torch.full((81, 81), -1e9)
    for i in range(81):
        r1, c1 = i // 9, i % 9
        for j in range(81):
            r2, c2 = j // 9, j % 9
            # Same row or column
            if r1 == r2 or c1 == c2:
                mask[i, j] = 0
            # Same diagonal
            elif abs(r1 - r2) == abs(c1 - c2):
                mask[i, j] = 0
    return mask

class MultiHeadShogiAttention(nn.Module):
    def __init__(self, dim, heads=8, type='square'):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.type = type
        self.head_dim = dim // heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        
        if type == 'ray':
            self.register_buffer('ray_mask', get_shogi_ray_mask())

    def forward(self, x, mask=None, piece_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if self.type == 'ray':
            # ray_mask is [81, 81], we need [B, heads, 81, 81]
            attn = attn + self.ray_mask[:N, :N].view(1, 1, N, N)
        elif self.type == 'piece' and piece_mask is not None:
            # piece_mask: [B, N, N]
            attn = attn + piece_mask.unsqueeze(1)
            
        if mask is not None:
            # mask: [B, N, 1] -> [B, 1, 1, N]
            m = mask.view(B, 1, 1, N).expand(-1, self.heads, N, -1)
            attn = attn.masked_fill(m == 0, -1e9)
            
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)

class ShogiBoardTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.sq_attn = MultiHeadShogiAttention(dim, heads, type='square')
        self.ray_attn = MultiHeadShogiAttention(dim, heads, type='ray')
        self.piece_attn = MultiHeadShogiAttention(dim, heads, type='piece')
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, x, mask=None, piece_mask=None):
        x = x + self.sq_attn(self.norm1(x), mask)
        x = x + self.ray_attn(self.norm2(x), mask)
        x = x + self.piece_attn(self.norm3(x), mask, piece_mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x

# ============================================================
# 4. Move Transformer
# ============================================================
class MoveTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, move_tokens, board_tokens):
        # move_tokens: [B, M, D], board_tokens: [B, 81, D]
        attn_out, _ = self.cross_attn(self.norm1(move_tokens), board_tokens, board_tokens)
        move_tokens = move_tokens + attn_out
        move_tokens = move_tokens + self.ffn(self.norm2(move_tokens))
        return move_tokens

# ============================================================
# 5. Temporal Transformer
# ============================================================
class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads=8, layers=4):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x):
        return self.enc(x)

# ============================================================
# 6. Full Yugiwarabe Network
# ============================================================
class YugiwarabeNet(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_dim=256):
        super().__init__()
        # 1. Local Feature Extractor
        self.cnn = CNNStem(input_channels, hidden_dim, num_blocks=4)
        
        # 2. GNN (Squares + Hand)
        # 81 squares + 14 hand pieces (7 types * 2 players) = 95 tokens?
        # Alternatively, simplify hand as part of history or special nodes.
        self.gnn_blocks = nn.ModuleList([FlashGNNLayer(hidden_dim) for _ in range(4)])
        
        # 3. Board Transformer
        self.board_transformer = nn.ModuleList([
            ShogiBoardTransformerBlock(hidden_dim) for _ in range(8)
        ])
        
        # 4. Move Transformer
        # We need a way to embed moves. Move tokens are usually legal moves.
        # For simplicity in this base, we'll implement the main board logic.
        self.move_transformer = nn.ModuleList([
            MoveTransformerBlock(hidden_dim) for _ in range(4)
        ])
        
        # 5. Temporal Transformer
        self.temporal_transformer = TemporalTransformer(hidden_dim, layers=4)
        
        # Positional Embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, 81, hidden_dim) * 0.02)
        
        # 7. Multi-task Heads
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Aux Heads
        self.aux_heads = nn.ModuleDict({
            "king_safety": nn.Linear(hidden_dim, 1),
            "attack": nn.Linear(hidden_dim, 1),
            "material": nn.Linear(hidden_dim, 1),
            "mobility": nn.Linear(hidden_dim, 1),
            "threat": nn.Linear(hidden_dim, 1),
            "damage": nn.Linear(hidden_dim, 1)
        })

    def forward(self, x_seq, piece_mask=None):
        # x_seq: [B, T, C, 9, 9]
        B, T, C, H, W = x_seq.shape
        x = x_seq.view(B * T, C, H, W)
        
        # 1. CNN
        feat = self.cnn(x) # [BT, D, 9, 9]
        feat = feat.view(B * T, -1, 81).permute(0, 2, 1) # [BT, 81, D]
        feat = feat + self.pos_emb
        
        # 2. GNN
        for gnn in self.gnn_blocks:
            feat = gnn(feat)
            
        # 3. Board Transformer
        # piece_mask: [B*T, 81, 81] should identify squares with same piece types
        for block in self.board_transformer:
            feat = block(feat, piece_mask=piece_mask)
            
        # 4. Temporal reasoning
        pooled = feat.view(B, T, 81, -1).mean(dim=2) # [B, T, D]
        history_context = self.temporal_transformer(pooled)
        last_context = history_context[:, -1, :]
        
        # Heads
        policy = self.policy_head(last_context)
        value = self.value_head(last_context)
        
        aux_results = {k: head(last_context) for k, head in self.aux_heads.items()}
        
        return F.log_softmax(policy, dim=-1), value, aux_results

def create_yugiwarabe(input_channels, num_actions):
    return YugiwarabeNet(input_channels, num_actions)
