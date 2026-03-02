import sys
import cshogi
from search_clab_gpu import GPUSearchClab

def main():
    """
    USI Protocol implementation for Cl-ab v2.0 AI.
    """
    board = cshogi.Board()
    searcher = GPUSearchClab()
    
    while True:
        line = sys.stdin.readline()
        if not line:
            break
            
        line = line.strip()
        if line == "usi":
            print("id name Cl-ab-v2-GPU-Complete")
            print("id author Shogi-AI-Team")
            print("usiok")
        elif line == "isready":
            print("readyok")
        elif line == "usinewgame":
            pass
        elif line.startswith("position"):
            # position [startpos | sfen <sfenstring>] [moves <move1> ... <moveN>]
            parts = line.split()
            if "startpos" in parts:
                board.reset()
            elif "sfen" in parts:
                # sfen is exactly 4 parts plus potentially 'moves'
                sfen_index = parts.index("sfen")
                sfen_str = " ".join(parts[sfen_index+1:sfen_index+5])
                board.set_sfen(sfen_str)
                
            if "moves" in parts:
                moves_index = parts.index("moves")
                for move_usi in parts[moves_index+1:]:
                    board.push_usi(move_usi)
        elif line.startswith("go"):
            # Execute Frontier-based Search
            best_move = searcher.search(board, max_depth=3)
            print(f"bestmove {best_move}")
        elif line == "quit":
            break

if __name__ == "__main__":
    main()
