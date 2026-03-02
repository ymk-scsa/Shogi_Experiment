import torch

# Piece Constants (match board_gpu.py)
BLACK, WHITE = 0, 1

# Material values
PIECE_VALUES = [
    0, 100, 300, 400, 500, 600, 800, 1000, 10000, # 0-8
    700, 700, 700, 700, 1100, 1300              # 9-14 (Promoted)
]
HAND_VALUES = [100, 300, 400, 500, 600, 800, 1000] # PAWN to ROOK

class GPUEvaluator:
    """
    Evaluates Shogi positions for the Cl-ab GPU-Complete Engine.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.v_piece = torch.tensor(PIECE_VALUES, dtype=torch.int32, device=self.device)
        self.v_hand = torch.tensor(HAND_VALUES, dtype=torch.int32, device=self.device)

    def evaluate_batch(self, piece_bb, hands, turns):
        """
        piece_bb: (N, 2, 15, 81) bool
        hands: (N, 2, 7) uint8
        turns: (N,) uint8
        """
        N = piece_bb.shape[0]
        
        # 1. Square evaluation (N, 2, 15) -> (N, 2)
        counts = piece_bb.sum(dim=3).to(torch.int32)
        score_board = (counts * self.v_piece).sum(dim=2)
        
        # 2. Hand evaluation (N, 2, 7) -> (N, 2)
        score_hand = (hands.to(torch.int32) * self.v_hand).sum(dim=2)
        
        # 3. Total PERSPECTIVE score (Relative to turn)
        total_black = score_board[:, BLACK] + score_hand[:, BLACK]
        total_white = score_board[:, WHITE] + score_hand[:, WHITE]
        
        diff = total_black - total_white
        # turn == BLACK(0) -> diff, turn == WHITE(1) -> -diff
        scores = torch.where(turns == BLACK, diff, -diff)
        
        return scores
