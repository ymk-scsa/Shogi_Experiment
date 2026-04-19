import numpy as np
import cshogi
from cshogi import HuffmanCodedPosAndEval


class HcpeDataWriter:
    def __init__(self) -> None:
        self.records = []
        self.board = cshogi.Board()

    def push(self, move: str) -> None:
        # 盤面をhcp形式に変換
        hcp = np.zeros(32, dtype=np.uint8)
        self.board.to_hcp(hcp)

        # usiの指し手を16bit moveに変換
        move16 = self.board.move_from_usi(move)

        self.records.append((hcp.copy(), move16))

        # 手を進める
        self.board.push_usi(move)

    def finalize(self, winner: int, filename: str) -> None:
        if winner not in (-1, 0, 1):
            raise ValueError("winner must be -1, 0, or 1")

        hcpe_array = np.zeros(len(self.records), HuffmanCodedPosAndEval)

        for i, (hcp, move16) in enumerate(self.records):
            hcpe_array[i]["hcp"] = hcp
            hcpe_array[i]["bestMove16"] = move16
            hcpe_array[i]["gameResult"] = winner
            hcpe_array[i]["eval"] = 0
            hcpe_array[i]["dummy"] = 0

        # 追記モードで開いて保存
        with open(filename, "ab") as f:
            hcpe_array.tofile(f)

    def reset(self, sfen: str = "startpos", moves: list[str] = []) -> None:
        if sfen == "startpos":
            sfen = "lnsgkgsnl/1r5b1/p1ppppppp/9/9/9/P1PPPPPPP/1B5R1/LNSGKGSNL b - 1"
        self.board.set_sfen(sfen)
        for move in moves:
            self.board.push_usi(move)
        self.records.clear()
