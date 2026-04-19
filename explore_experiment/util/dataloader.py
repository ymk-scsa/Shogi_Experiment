import os
from typing import Callable, Union, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import torch

from cshogi import Board, HuffmanCodedPosAndEval
from shogi.feature import FEATURES_NUM, make_input_features, make_move_label, make_result

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
