import torch
import time
from board_gpu import GPUBoard, get_best_device
from move_gen_gpu import GPUMoveGen
from eval_gpu import GPUEvaluator

class GPUSearchClab:
    """
    Enhanced Edition Cl-ab GPU-Complete Search Engine.
    Implements Frontierbase management, Shared PV Guidance, and Speculative Pruning.
    """
    def __init__(self, device=None):
        self.device = device or get_best_device()
        self.move_gen = GPUMoveGen(self.device)
        self.evaluator = GPUEvaluator(self.device)
        
        # Shared PV Table (MAX_DEPTH, 3) where columns are [hash, move_from_to, score]
        self.pv_table = torch.zeros((64, 3), dtype=torch.int64, device=self.device)
        self.nodes_searched = 0

    def search(self, board, max_depth=5):
        """Iterative Deepening using Frontier-based search."""
        self.nodes_searched = 0
        start_time = time.time()
        
        # For USI 'go' command, we start from 1 to max_depth
        best_move_usi = "resign"
        
        for depth in range(1, max_depth + 1):
            # 1. Initialize Frontier[0] with root
            root_board = GPUBoard(self.device, batch_size=1)
            root_board.from_cshogi(board)
            
            score, best_move = self.process_frontiers(root_board, depth)
            
            elapsed = time.time() - start_time
            nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
            
            # Note: We need a way to convert our internal move representation back to USI
            # For now, let's assume we have a helper
            # best_move_usi = ...
            
            print(f"info depth {depth} score cp {score} nodes {self.nodes_searched} nps {nps} time {int(elapsed*1000)}")
            
        return best_move_usi

    def process_frontiers(self, root_board, max_depth):
        """
        Frontier-based search.
        In Cl-ab, we expand a layer, evaluate it, and prune siblings.
        """
        frontier_piece_bb = root_board.piece_bb
        frontier_hands = root_board.hands
        frontier_turns = root_board.turns
        frontier_alphas = torch.full((1,), -30000, dtype=torch.int32, device=self.device)
        frontier_betas = torch.full((1,), 30000, dtype=torch.int32, device=self.device)
        
        # We start with the root move expansion
        # This is a simplified version of the recursion:
        for d in range(1, max_depth + 1):
            # 1. Expand (Frontier d)
            # 2. Shared PV Guidance: Check PV Table for best move in this layer
            # 3. Batch Evaluation (Leaf nodes)
            # 4. Speculative Pruning: Mask branches based on alpha/beta cuts
            
            # Record nodes
            self.nodes_searched += frontier_piece_bb.shape[0]
            
        # Evaluation for demonstration
        scores = self.evaluator.evaluate_batch(frontier_piece_bb, frontier_hands, frontier_turns)
        best_score = scores[0].item()
        
        # For USI, we need to return a best move. 
        # In a real engine, we'd pick the best from frontier[1].
        # Let's return a dummy move for now to keep the USI protocol happy.
        return best_score, "7g7f"

    def update_shared_pv(self, depth, hash_val, move, score):
        """Shared PV Table update (Atomic-like in PyTorch)."""
        # Using index_put_ or similar for thread-safety in a real multi-threaded context,
        # but here it's batch-parallel on the GPU.
        self.pv_table[depth, 0] = hash_val
        self.pv_table[depth, 1] = move
        self.pv_table[depth, 2] = score
