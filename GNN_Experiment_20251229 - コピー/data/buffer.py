"""
data/buffer.py
HCPEフォーマットおよびPSV(PackedSfenValue)フォーマットの棋譜データを読み込み、
ミニバッチを生成するデータローダー。
shogiAI の dataloader.py を GNN_Experiment_20251229 用に移植。
"""

import os
import sys
import logging
import math
from typing import Union, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from cshogi import Board, HuffmanCodedPosAndEval, PackedSfenValue, move16_from_psv, BLACK

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game.board import FEATURES_SETTINGS, make_move_label, make_result

# モジュールレベルロガー
_logger = logging.getLogger(__name__)

# PSVのvalue変換パラメータ
_PSV_SCORE_SCALE = 600.0   # sigmoid のスケール係数（600cp ≈ 勝率73%）


class HcpeDataLoader:
    """
    HuffmanCodedPosAndEval (HCPE) フォーマットの棋譜ファイルを読み込み、
    PyTorch テンソルのミニバッチを非同期で供給するデータローダー。
    """

    def __init__(
        self,
        files:        Union[list, tuple, str],
        batch_size:   int,
        device:       torch.device,
        shuffle:      bool         = False,
        features_mode: int         = 0,
        limit:        Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.device     = device
        self.shuffle    = shuffle
        self.limit      = limit

        self.load(files)

        self.features_settings = FEATURES_SETTINGS[features_mode]
        ch = self.features_settings.features_num

        pin = device.type != "cpu"
        self.torch_features   = torch.empty((batch_size, ch, 9, 9), dtype=torch.float32, pin_memory=pin)
        self.torch_move_label = torch.empty((batch_size,),           dtype=torch.int64,   pin_memory=pin)
        self.torch_result     = torch.empty((batch_size, 1),         dtype=torch.float32, pin_memory=pin)

        # numpy ビュー（ゼロコピー）
        self.features   = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result     = self.torch_result.numpy().reshape(-1)

        self.i = 0
        self.l = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.board    = Board()

    def load(self, files: Union[list, tuple, str]) -> None:
        """HCPE ファイルを読み込んで self.data に numpy 配列として保存する。"""
        if isinstance(files, str):
            files = [files]
        data = []
        for path in files:
            if os.path.exists(path):
                _logger.info("Loading: %s", path)
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                _logger.warning("%s not found, skipping", path)

        if not data:
            raise FileNotFoundError("HCPE data not found: {}".format(files))

        self.data = np.concatenate(data)

        if self.limit is not None:
            if self.shuffle:
                np.random.shuffle(self.data)
            self.data = self.data[: self.limit]

    def mini_batch(
        self, hcpevec: np.ndarray
    ) -> tuple:
        """hcpe 配列からひとつのミニバッチを作成して Tensor のタプルを返す。"""
        self.features.fill(0)
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe["hcp"])
            self.features_settings.make_features(self.board, self.features[i])
            self.move_label[i] = make_move_label(hcpe["bestMove16"], self.board.turn)
            self.result[i]     = make_result(hcpe["gameResult"], self.board.turn)

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

    def sample(self) -> tuple:
        """データからランダムに batch_size 個サンプリングしてミニバッチを返す。"""
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self) -> None:
        """次のバッチを非同期で事前取得する。"""
        hcpevec = self.data[self.i: self.i + self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            _logger.debug("Reached end of data (len=%d < batch_size=%d)", len(hcpevec), self.batch_size)
            return
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

    def __next__(self) -> tuple:
        if self.l > len(self.data):
            raise StopIteration()
        result = self.f.result()
        self.l = self.i
        self.pre_fetch()
        return result


# ---------------------------------------------------------------------------
# PSV (PackedSfenValue) データローダー
# ---------------------------------------------------------------------------

def _score_to_value(score: np.ndarray, turn: int) -> float:
    """
    エンジン評価値 score (先手視点 cp) を value 教師 [0, 1] に変換する。

    sigmoid(score / scale) で変換し、
    手番が後手 (turn == WHITE) の場合は 1 - value で反転する。
    """
    v = 1.0 / (1.0 + math.exp(-float(score) / _PSV_SCORE_SCALE))
    if turn != BLACK:
        v = 1.0 - v
    return v


class PsvDataLoader:
    """
    PackedSfenValue (PSV) フォーマットの棋譜ファイルを読み込み、
    PyTorch テンソルのミニバッチを非同期で供給するデータローダー。

    HcpeDataLoader と同一インターフェース（features / move_label / result）を提供する。

    value 教師の変換:
        score (int16, 先手視点 cp) → sigmoid(score / 600) → [0, 1]
        後手手番の場合は 1 - value で反転。
    """

    def __init__(
        self,
        files:         Union[list, tuple, str],
        batch_size:    int,
        device:        torch.device,
        shuffle:       bool          = False,
        features_mode: int           = 0,
        limit:         Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.device     = device
        self.shuffle    = shuffle
        self.limit      = limit

        self.load(files)

        self.features_settings = FEATURES_SETTINGS[features_mode]
        ch = self.features_settings.features_num

        pin = device.type != "cpu"
        self.torch_features   = torch.empty((batch_size, ch, 9, 9), dtype=torch.float32, pin_memory=pin)
        self.torch_move_label = torch.empty((batch_size,),           dtype=torch.int64,   pin_memory=pin)
        self.torch_result     = torch.empty((batch_size, 1),         dtype=torch.float32, pin_memory=pin)

        # numpy ビュー（ゼロコピー）
        self.features   = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        self.result     = self.torch_result.numpy().reshape(-1)

        self.i = 0
        self.l = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.board    = Board()

    def load(self, files: Union[list, tuple, str]) -> None:
        """PSV ファイルを読み込んで self.data に numpy 配列として保存する。"""
        if isinstance(files, str):
            files = [files]
        data = []
        for path in files:
            if os.path.exists(path):
                _logger.info("Loading PSV: %s", path)
                data.append(np.fromfile(path, dtype=PackedSfenValue))
            else:
                _logger.warning("%s not found, skipping", path)

        if not data:
            raise FileNotFoundError("PSV data not found: {}".format(files))

        self.data = np.concatenate(data)

        if self.limit is not None:
            if self.shuffle:
                np.random.shuffle(self.data)
            self.data = self.data[: self.limit]

    def mini_batch(self, psvvec: np.ndarray) -> tuple:
        """psv 配列からひとつのミニバッチを作成して Tensor のタプルを返す。"""
        self.features.fill(0)
        for i, psv in enumerate(psvvec):
            # 盤面の復元
            self.board.set_psfen(psv["sfen"])
            # 入力特徴量
            self.features_settings.make_features(self.board, self.features[i])
            # 指し手ラベル（PSV独自エンコード→通常move16→ラベルインデックス）
            move16 = move16_from_psv(int(psv["move"]))
            self.move_label[i] = make_move_label(move16, self.board.turn)
            # value 教師（score → sigmoid）
            self.result[i] = _score_to_value(psv["score"], self.board.turn)

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

    def sample(self) -> tuple:
        """データからランダムに batch_size 個サンプリングしてミニバッチを返す。"""
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))

    def pre_fetch(self) -> None:
        """次のバッチを非同期で事前取得する。"""
        hcpevec = self.data[self.i: self.i + self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            _logger.debug("Reached end of data (len=%d < batch_size=%d)", len(hcpevec), self.batch_size)
            return
        self.f = self.executor.submit(self.mini_batch, hcpevec)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> "PsvDataLoader":
        self.i = 0
        self.l = 0
        if self.shuffle:
            np.random.shuffle(self.data)
        self.pre_fetch()
        return self

    def __next__(self) -> tuple:
        if self.l > len(self.data):
            raise StopIteration()
        result = self.f.result()
        self.l = self.i
        self.pre_fetch()
        return result
