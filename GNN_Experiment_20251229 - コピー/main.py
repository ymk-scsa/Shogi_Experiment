"""
agent/main.py
USI（将棋所/Shogidokoro）通信エンジンのエントリーポイント。
shogiAI の base_player.py の run() ループを移植し、
MCTSPlayer（search/mcts.py）を呼び出す。
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

from search.mcts import MCTSPlayer

# USI コマンドの戻り値の型エイリアス (bestmove, ponder_move)
UsiResponse = tuple


def run(player: MCTSPlayer) -> None:
    """
    USI 通信ループ。
    標準入力からコマンドを1行ずつ読み込み、対応するメソッドを呼び出す。
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future: Optional[Future] = None
    last_limits: dict = {}

    while True:
        try:
            cmd_line = input().strip()
        except EOFError:
            break

        if not cmd_line:
            continue

        cmd = cmd_line.split(" ", 1)

        if cmd[0] == "usi":
            player.usi()
            print("usiok", flush=True)

        elif cmd[0] == "setoption":
            option = cmd[1].split(" ")
            player.setoption(option)

        elif cmd[0] == "isready":
            player.isready()
            print("readyok", flush=True)

        elif cmd[0] == "usinewgame":
            pass  # 新局の初期化はisready時に済んでいる

        elif cmd[0] == "position":
            args = cmd[1].split("moves")
            sfen = args[0].strip()
            usi_moves = args[1].split() if len(args) > 1 else []
            player.position(sfen, usi_moves)

        elif cmd[0] == "go":
            kwargs: dict = {}
            if len(cmd) > 1:
                args = cmd[1].split(" ")
                if args[0] == "infinite":
                    kwargs["infinite"] = True
                else:
                    if args[0] == "ponder":
                        kwargs["ponder"] = True
                        args = args[1:]
                    for i in range(0, len(args) - 1, 2):
                        key = args[i]
                        if key == "btime":
                            kwargs["btime"] = int(args[i + 1])
                        elif key == "wtime":
                            kwargs["wtime"] = int(args[i + 1])
                        elif key == "byoyomi":
                            kwargs["byoyomi"] = int(args[i + 1])
                        elif key == "binc":
                            kwargs["binc"] = int(args[i + 1])
                        elif key == "winc":
                            kwargs["winc"] = int(args[i + 1])
                        elif key == "nodes":
                            kwargs["nodes"] = int(args[i + 1])

            player.set_limits(**kwargs)
            last_limits = dict(kwargs)
            need_print_bestmove = "ponder" not in kwargs and "infinite" not in kwargs

            def go_and_print_bestmove() -> UsiResponse:
                bestmove, ponder_move = player.go()
                if need_print_bestmove:
                    print(
                        "bestmove " + bestmove
                        + (" ponder " + ponder_move if ponder_move else ""),
                        flush=True,
                    )
                return bestmove, ponder_move

            future = executor.submit(go_and_print_bestmove)

        elif cmd[0] == "stop":
            player.stop()
            if future is not None:
                bestmove, _ = future.result()
                print("bestmove " + bestmove, flush=True)

        elif cmd[0] == "ponderhit":
            last_limits["ponder"] = False
            player.ponderhit(last_limits)
            if future is not None:
                bestmove, ponder_move = future.result()
                print(
                    "bestmove " + bestmove
                    + (" ponder " + ponder_move if ponder_move else ""),
                    flush=True,
                )

        elif cmd[0] == "quit":
            player.quit()
            if future is not None:
                future.result()
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="30blocks", help="Model architecture mode (modelA, modelB, etc.)")
    parser.add_argument("--weights", type=str, default="weights/checkpoint.pth", help="Path to weights file")
    parser.add_argument("--name", type=str, default="GNN-Shogi", help="USI engine name")
    args = parser.parse_args()

    # 引数に基づいて MCTSPlayer を起動
    player = MCTSPlayer(
        features_mode=0,         # 0: standard (46ch)
        blocks_config_mode=args.mode,
        name=args.name,
        modelfile=args.weights,
    )
    run(player)

