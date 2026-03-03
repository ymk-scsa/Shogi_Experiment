import sys
import os

# Root of Shogi_Experience
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yugiwarabe.search.mcts import MCTSPlayer

def run(player: MCTSPlayer):
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            cmd = line.strip().split()
            if not cmd: continue

            if cmd[0] == "usi":
                player.usi()
            elif cmd[0] == "isready":
                player.isready()
            elif cmd[0] == "position":
                idx = cmd.index("moves") if "moves" in cmd else len(cmd)
                sfen = " ".join(cmd[1:idx])
                moves = cmd[idx+1:] if idx < len(cmd) else []
                player.position(sfen, moves)
            elif cmd[0] == "go":
                kwargs = {}
                if "nodes" in cmd:
                    kwargs["nodes"] = cmd[cmd.index("nodes") + 1]
                player.set_limits(**kwargs)
                best, ponder = player.go()
                print(f"bestmove {best}{' ponder ' + ponder if ponder else ''}")
                sys.stdout.flush()
            elif cmd[0] == "quit":
                break
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/yugiwarabe.pth")
    args = parser.parse_args()

    player = MCTSPlayer(modelfile=args.weights)
    run(player)
