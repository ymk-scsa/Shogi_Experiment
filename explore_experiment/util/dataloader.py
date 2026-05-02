import math
import os
from typing import Callable, Union, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import torch

from cshogi import Board, BLACK, HuffmanCodedPosAndEval, PackedSfenValue, move16_from_psv
from shogi.feature import FEATURES_NUM, make_input_features, make_move_label, make_result
from shogi.halfkav2_feature import extract_halfkav2_indices

from util.logger import Logger


class HcpeDataLoader:
    def __init__(
        self,
        files: Union[list[str], tuple[str], str],
        batch_size: int,
        device: torch.device,
        shuffle: bool = False,
        features_num: int = FEATURES_NUM,
        make_features: Callable[[Board, np.ndarray], None] = make_input_features,
        limit: Optional[int] = None,
    ) -> None:
        self.logging = Logger("hcpe dataloder").get_logger()
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.limit = limit
        self.make_features = make_features
        self.load(files)

        self.torch_features = torch.empty(
            (batch_size, features_num, 9, 9),
            dtype=torch.float32,
            pin_memory=device.type != "cpu",
        )
        self.torch_move_label = torch.empty((batch_size), dtype=torch.int64, pin_memory=device.type != "cpu")
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=device.type != "cpu")

        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)

        self.i = 0
        self.l = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.board = Board()

    def load(self, files: Union[list[str], tuple[str], str]) -> None:
        data = []
        if isinstance(files, str):
            files = [files]
        for path in files:
            if os.path.exists(path):
                logging.info(path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                logging.warn("{} not found, skipping".format(path))

        self.data = np.concatenate(data)

        if self.limit is not None:
            if self.shuffle:
                np.random.shuffle(self.data)
            self.data = self.data[: self.limit]

    def mini_batch(self, hcpevec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.features.fill(0)
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe["hcp"])  # ボードを設定
            self.make_features(self.board, self.features[i])  # 入力特徴量の作成
            self.move_label[i] = make_move_label(hcpe["bestMove16"], self.board.turn)  # 正解データ方策
            self.result[i] = make_result(hcpe["gameResult"], self.board.turn)  # 正解データ価値

        if self.device.type == "cpu":
            return (
                self.torch_features.clone(),
                self.torch_move_label.clone(),
                self.torch_result.clone(),
            )
        else:
            return (
                self.torch_features.to(self.device),
                self.torch_move_label.to(self.device),
                self.torch_result.to(self.device),
            )

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self) -> None:
        hcpevec = self.data[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            self.logging.debug("len(hcpevec) < self.batch_size")
            return
        # if len(hcpevec) <= 0:
        #    return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> "HcpeDataLoader":
        self.i = 0
        self.l = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.l > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.l = self.i
        self.pre_fetch()

        return result
        # dlshogi 1/6


def _psv_score_to_value(score: object, turn: int, scale: float) -> float:
    """PSV の score（先手視点 cp 想定）を value 教師 [0, 1] に変換する。"""
    s = float(score)
    v = 1.0 / (1.0 + math.exp(-s / scale))
    if turn != BLACK:
        v = 1.0 - v
    return float(v)


class PsvDataLoader:
    """PackedSfenValue (PSV, psv.bin) を読み、HcpeDataLoader と同じインターフェースでバッチ化する。"""

    def __init__(
        self,
        files: Union[list[str], tuple[str], str],
        batch_size: int,
        device: torch.device,
        shuffle: bool = False,
        features_num: int = FEATURES_NUM,
        make_features: Callable[[Board, np.ndarray], None] = make_input_features,
        limit: Optional[int] = None,
        score_scale: float = 600.0,
    ) -> None:
        self.logging = Logger("psv dataloder").get_logger()
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.limit = limit
        self.score_scale = score_scale
        self.make_features = make_features
        self.load(files)

        self.torch_features = torch.empty(
            (batch_size, features_num, 9, 9),
            dtype=torch.float32,
            pin_memory=device.type != "cpu",
        )
        self.torch_move_label = torch.empty((batch_size), dtype=torch.int64, pin_memory=device.type != "cpu")
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=device.type != "cpu")

        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)

        self.i = 0
        self.l = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.board = Board()

    def load(self, files: Union[list[str], tuple[str], str]) -> None:
        data = []
        if isinstance(files, str):
            files = [files]
        for path in files:
            if os.path.exists(path):
                logging.info(path)
                data.append(np.fromfile(path, dtype=PackedSfenValue))
            else:
                logging.warn("{} not found, skipping".format(path))

        if not data:
            raise FileNotFoundError(f"PSV file(s) not found or empty: {files}")

        self.data = np.concatenate(data)

        if self.limit is not None:
            if self.shuffle:
                np.random.shuffle(self.data)
            self.data = self.data[: self.limit]

    def mini_batch(self, psvvec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.features.fill(0)
        for i, psv in enumerate(psvvec):
            self.board.set_psfen(psv["sfen"])
            self.make_features(self.board, self.features[i])
            move16 = move16_from_psv(int(psv["move"]))
            self.move_label[i] = make_move_label(move16, self.board.turn)
            self.result[i] = _psv_score_to_value(psv["score"], self.board.turn, self.score_scale)

        if self.device.type == "cpu":
            return (
                self.torch_features.clone(),
                self.torch_move_label.clone(),
                self.torch_result.clone(),
            )
        else:
            return (
                self.torch_features.to(self.device),
                self.torch_move_label.to(self.device),
                self.torch_result.to(self.device),
            )

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self) -> None:
        psvvec = self.data[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        if len(psvvec) < self.batch_size:
            self.logging.debug("len(psvvec) < self.batch_size")
            return

        self.f = self.executor.submit(self.mini_batch, psvvec)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> "PsvDataLoader":
        self.i = 0
        self.l = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.l > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.l = self.i
        self.pre_fetch()

        return result


class HcpeNnueDataLoader:
    """HCPE を読み、HalfKAv2 疎インデックス + 教師（方策ラベル・勝敗）をバッチ化する。"""

    def __init__(
        self,
        files: Union[list[str], tuple[str], str],
        batch_size: int,
        device: torch.device,
        shuffle: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self.logging = Logger("hcpe nnue dataloader").get_logger()
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.limit = limit
        self.load(files)

        pin = device.type != "cpu"
        self.torch_move_label = torch.empty((batch_size,), dtype=torch.int64, pin_memory=pin)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=pin)
        self.move_label = self.torch_move_label.numpy()
        self.result = self.torch_result.numpy().reshape(-1)

        self.i = 0
        self.l = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.board = Board()

    def load(self, files: Union[list[str], tuple[str], str]) -> None:
        data = []
        if isinstance(files, str):
            files = [files]
        for path in files:
            if os.path.exists(path):
                logging.info(path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                logging.warn("{} not found, skipping".format(path))

        self.data = np.concatenate(data)

        if self.limit is not None:
            if self.shuffle:
                np.random.shuffle(self.data)
            self.data = self.data[: self.limit]

    def mini_batch(self, hcpevec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_parts: list[torch.Tensor] = []
        offsets_list: list[int] = [0]

        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe["hcp"])
            idx = extract_halfkav2_indices(self.board, self.board.turn)
            t = torch.from_numpy(idx).long()
            flat_parts.append(t)
            offsets_list.append(offsets_list[-1] + int(t.numel()))
            self.move_label[i] = make_move_label(hcpe["bestMove16"], self.board.turn)
            self.result[i] = make_result(hcpe["gameResult"], self.board.turn)

        flat = torch.cat(flat_parts, dim=0)
        # EmbeddingBag: offsets[i] は i 番目のバッグの先頭（長さ batch_size）
        offsets = torch.tensor(offsets_list[:-1], dtype=torch.long)

        if self.device.type == "cpu":
            return (
                flat.clone(),
                offsets.clone(),
                self.torch_move_label.clone(),
                self.torch_result.clone(),
            )
        return (
            flat.to(self.device),
            offsets.to(self.device),
            self.torch_move_label.to(self.device),
            self.torch_result.to(self.device),
        )

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self) -> None:
        hcpevec = self.data[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            self.logging.debug("len(hcpevec) < self.batch_size")
            return

        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> "HcpeNnueDataLoader":
        self.i = 0
        self.l = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.l > len(self.data):
            raise StopIteration()

        result = self.f.result()
        self.l = self.i
        self.pre_fetch()

        return result
