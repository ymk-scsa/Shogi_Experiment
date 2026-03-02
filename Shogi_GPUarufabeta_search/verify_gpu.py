import torch
import cshogi
from board_gpu import GPUBoard, get_best_device
from eval_gpu import GPUEvaluator
from search_clab_gpu import GPUSearchClab

def test_clab_architecture():
    device = get_best_device()
    print(f"Testing Cl-ab Architecture on: {device}")
    
    # 1. Test GPUBoard Batch + Zobrist
    board = cshogi.Board()
    gpuboard = GPUBoard(device, batch_size=2)
    gpuboard.from_cshogi(board, batch_idx=0)
    
    # Push a move on cshogi and load to second batch slot
    board.push_usi("7g7f")
    gpuboard.from_cshogi(board, batch_idx=1)
    
    print("Hashes:", gpuboard.hashes)
    assert gpuboard.hashes[0] != gpuboard.hashes[1], "Zobrist hashes must change after move!"
    print("GPUBoard batching successful.")

    # 2. Test Batch Evaluation
    evaluator = GPUEvaluator(device)
    scores = evaluator.evaluate_batch(gpuboard.piece_bb, gpuboard.hands, gpuboard.turns)
    print("Scores:", scores)
    assert scores.shape[0] == 2, "Batch evaluation should return 2 scores."
    print("Batch evaluation successful.")

    # 3. Test Cl-ab Search integration
    searcher = GPUSearchClab(device)
    board.reset()
    print("Running Frontier Search (depth 3)...")
    best_move = searcher.search(board, max_depth=3)
    print(f"Best move: {best_move}")
    assert isinstance(best_move, str), "Search should return a USI move string."
    print("Cl-ab search integration successful.")

if __name__ == "__main__":
    try:
        test_clab_architecture()
        print("\nAll Cl-ab tests passed!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
