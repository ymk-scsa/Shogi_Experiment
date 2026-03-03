import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. Flash Interaction Layer (from Yu-Gi-Oh! FlashGNN)
# ============================================================
class FlashGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.msg_proj = nn.Linear(dim, dim)
        self.update_proj = nn.Linear(dim * 2, dim)

    def forward(self, x, mask):
        # x: [B, N, D], mask: [B, N, 1]
        masked_x = x * mask
        mean_msg = masked_x.sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        mean_msg = mean_msg.expand_as(x)

        h = torch.cat([x, self.msg_proj(mean_msg)], dim=-1)
        return F.relu(self.update_proj(h))

class LightSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x, mask):
        # x: [B, N, D], mask: [B, N, 1]
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(x.size(-1))

        # mask: [B, N, 1] -> [B, 1, N]
        attn_mask = mask.transpose(1, 2)
        scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)

# ============================================================
# 2. Board Encoder (Hybrid GNN + Attention)
# ============================================================
class BoardEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.gcn = FlashGCNLayer(hidden_dim)
        self.attn = LightSelfAttention(hidden_dim)
        
        # Positional encoding for 9x9 board
        self.pos_emb = nn.Parameter(torch.randn(1, 81, hidden_dim) * 0.02)

    def forward(self, x):
        # x: [B, C, 9, 9] -> [B, 81, C]
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        
        # mask based on presence of pieces (non-zero features)
        mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        
        h = self.input_proj(x)
        h = h + self.pos_emb
        
        # Flash interactions
        h = h + self.gcn(h, mask)
        h = h + self.attn(h, mask)
        
        # Global Pooling (Average)
        pooled = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return pooled, h

# ============================================================
# 3. Temporal Transformer (from Graformer)
# ============================================================
class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        # x: [B, T, D]
        return self.encoder(x, src_key_padding_mask=mask)

# ============================================================
# 4. Yugiwarabe Network (Fused Architecture)
# ============================================================
class YugiwarabeNet(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_dim=256, history_len=16):
        super().__init__()
        self.board_encoder = BoardEncoder(input_channels, hidden_dim)
        self.temporal_transformer = TemporalTransformer(embed_dim=hidden_dim)
        
        self.history_len = history_len
        self.pos_history_emb = nn.Parameter(torch.randn(1, history_len, hidden_dim) * 0.02)

        # Policy Head (Standard AlphaZero style)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # Value Head (Standard AlphaZero style)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, x_seq):
        """
        x_seq: [B, T, C, 9, 9] (History of board states)
        """
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        
        # Encode each board in the sequence
        board_embs, _ = self.board_encoder(x_flat)
        seq_embs = board_embs.view(B, T, -1)
        
        # Add history positional embedding
        if T <= self.history_len:
            seq_embs = seq_embs + self.pos_history_emb[:, :T, :]
        else:
            # Handle variable length if needed, here we assume it fits or truncated
            seq_embs = seq_embs + self.pos_history_emb[:, -T:, :]

        # Temporal attention over history
        history_context = self.temporal_transformer(seq_embs)
        
        # Last state context for decision
        last_context = history_context[:, -1, :]
        
        policy = self.policy_head(last_context)
        value = self.value_head(last_context)
        
        return F.log_softmax(policy, dim=1), value

def create_yugiwarabe(input_channels, num_actions):
    return YugiwarabeNet(input_channels, num_actions)
