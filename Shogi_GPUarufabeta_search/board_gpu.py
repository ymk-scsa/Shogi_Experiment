import torch
import numpy as np
import cshogi

def get_best_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# Shogi Piece Constants (cshogi mapping)
EMPTY = 0
PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK, KING = 1, 2, 3, 4, 5, 6, 7, 8
PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER, PRO_BISHOP, PRO_ROOK = 9, 10, 11, 12, 13, 14

BLACK, WHITE = 0, 1
PIECE_SYMBOLS = [" .", " P", " L", " N", " S", " G", " B", " R", " K", "+P", "+L", "+N", "+S", "+B", "+R"]

class GPUBoard:
    """
    Shogi Board optimized for Cl-ab GPU-complete search.
    Manages bitboards and Zobrist hashes in batches.
    """
    def __init__(self, device=None, batch_size=1):
        self.device = device or get_best_device()
        self.batch_size = batch_size
        
        # (N, color:2, piece_type:15, square:81)
        self.piece_bb = torch.zeros((batch_size, 2, 15, 81), dtype=torch.bool, device=self.device)
        self.hands = torch.zeros((batch_size, 2, 7), dtype=torch.uint8, device=self.device)
        self.turns = torch.zeros(batch_size, dtype=torch.uint8, device=self.device)
        
        # Zobrist Hashes (N,)
        self.hashes = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        self._init_zobrist()

    def _init_zobrist(self):
        """Initializes Zobrist tables on GPU."""
        # Square: 81, Color: 2, PieceType: 15
        torch.manual_seed(20260301)
        self.z_piece = torch.randint(-(2**63), 2**63 - 1, (81, 2, 15), dtype=torch.int64, device=self.device)
        self.z_hand = torch.randint(-(2**63), 2**63 - 1, (2, 7, 20), dtype=torch.int64, device=self.device)
        self.z_turn = torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=self.device)

    def from_cshogi(self, boards, batch_idx=0):
        """
        Updates batch entry from cshogi.Board(s).
        boards: a single cshogi.Board or a list.
        """
        if not isinstance(boards, list):
            boards = [boards]
            
        for i, b in enumerate(boards):
            idx = batch_idx + i
            if idx >= self.batch_size: break
            
            self.piece_bb[idx].fill_(False)
            self.hands[idx].fill_(0)
            self.turns[idx] = b.turn
            
            # Reset hash
            h = torch.tensor(0, dtype=torch.int64, device=self.device)
            if b.turn == WHITE: h ^= self.z_turn[0]
            
            for sq in range(81):
                p = b.piece(sq)
                if p != EMPTY:
                    pt = p & 0x0F
                    color = (p >> 4) & 0x01
                    self.piece_bb[idx, color, pt, sq] = True
                    h ^= self.z_piece[sq, color, pt]
            
            for color in range(2):
                for pt in range(1, 8):
                    count = b.pieces_in_hand[color][pt-1]
                    self.hands[idx, color, pt-1] = count
                    for c in range(count):
                        h ^= self.z_hand[color, pt-1, c]
            
            self.hashes[idx] = h

    def __str__(self):
        # Debug print for batch element 0
        bbs = self.piece_bb[0].cpu().numpy()
        res = ""
        for rank in range(9):
            row = ""
            for file in range(8, -1, -1):
                sq = rank * 9 + file
                pt, col = EMPTY, None
                for c in range(2):
                    for p in range(1, 15):
                        if bbs[c, p, sq]: pt, col = p, c; break
                sym = PIECE_SYMBOLS[pt]
                row += sym.lower() if col == WHITE else sym
            res += row + "\n"
        return res
