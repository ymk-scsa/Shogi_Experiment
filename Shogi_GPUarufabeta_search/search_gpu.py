import torch
import numpy as np
import time
import cshogi
from board_gpu import GPUBoard
from eval_gpu import GPUEvaluator

class GPUSearch:
    """
    Alpha-Beta search engine optimized for GPU.
    Uses CPU (cshogi) for move generation and GPU (PyTorch) for batch evaluation.
    """
    def __init__(self, board: GPUBoard):
        self.gpu_board = board
        self.device = board.device
        self.evaluator = GPUEvaluator(device=self.device)
        self.nodes_searched = 0
        self.max_depth = 3 # Default depth for pro-active search
        
        # We maintain a cshogi Board for move generation
        self.internal_board = cshogi.Board()

    def set_state(self, board: cshogi.Board):
        """Syncs the searcher's internal state with the given board."""
        self.internal_board.set_sfen(board.sfen())
        self.gpu_board.from_cshogi(self.internal_board)

    def search(self, depth=None):
        if depth is not None:
            self.max_depth = depth
        
        self.nodes_searched = 0
        start_time = time.time()
        
        best_move = None
        # Iterative Deepening
        for d in range(1, self.max_depth + 1):
            move, score = self.alpha_beta(self.internal_board, d, -30000, 30000)
            best_move = move
            elapsed = time.time() - start_time
            nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
            print(f"info depth {d} score cp {score} nodes {self.nodes_searched} nps {nps} time {int(elapsed*1000)}")
            
        return cshogi.to_usi(best_move).decode() if best_move else "resign"

    def alpha_beta(self, board, depth, alpha, beta):
        """Standard Alpha-Beta search using GPU for evaluation at depth 0."""
        self.nodes_searched += 1
        
        if board.is_game_over():
            return None, -20000 # Loss (simplistic)

        if depth <= 0:
            return None, self.evaluate_single(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, -20000
        
        # Batch Move Evaluation (Optimization)
        # Instead of recursing immediately, we evaluate all children at depth 1-ply
        if depth == 1:
            best_move, best_score = self.batch_evaluate_moves(board, legal_moves)
            return best_move, best_score

        best_move = None
        for move in legal_moves:
            board.push(move)
            _, score = self.alpha_beta(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()
            
            if score >= beta:
                return move, beta
            if score > alpha:
                alpha = score
                best_move = move
                
        return best_move, alpha

    def evaluate_single(self, board):
        """Evaluates a single board using GPU."""
        self.gpu_board.from_cshogi(board)
        piece_bbs, hands, turns = self.gpu_board.to_batch_tensors()
        score = self.evaluator.evaluate_batch(piece_bbs, hands, turns)
        return int(score[0])

    def batch_evaluate_moves(self, board, moves):
        """Evaluates a list of moves in a single GPU batch."""
        N = len(moves)
        piece_bbs = torch.zeros((N, 2, 15, 81), dtype=torch.bool, device=self.device)
        hands = torch.zeros((N, 2, 7), dtype=torch.int32, device=self.device)
        turns = torch.zeros(N, dtype=torch.int32, device=self.device)
        
        temp_board = cshogi.Board(board.sfen())
        gpu_temp = GPUBoard(device=self.device)
        
        for i, move in enumerate(moves):
            temp_board.push(move)
            gpu_temp.from_cshogi(temp_board)
            piece_bbs[i] = gpu_temp.piece_bb
            hands[i] = gpu_temp.hand
            turns[i] = temp_board.turn
            temp_board.pop()
            
        scores = self.evaluator.evaluate_batch(piece_bbs, hands, turns)
        scores = scores.cpu().numpy()
        
        best_idx = np.argmax(-scores)
        return moves[best_idx], int(-scores[best_idx])
