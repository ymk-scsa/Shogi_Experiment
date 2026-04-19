import os
import logging
from typing import Optional


class Logger:
    """アプリケーション全体で統一して利用するロガークラス"""

    def __init__(self, name: str, log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
        self.logger = logging.getLogger(name)

        # すでにハンドラーが追加されている場合は追加しない
        if not self.logger.hasHandlers():
            # 引数 > 環境変数 > デフォルト
            if log_level is None:
                log_level = os.getenv("LOG_LEVEL", "INFO").upper()

            # ログフォーマット
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

            # コンソール出力用のハンドラー
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # ファイル出力の設定（指定があれば）
            if log_file:
                file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            # 設定を適用
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))
            self.logger.addHandler(console_handler)
            self.logger.propagate = False  # ルートロガーへの伝播を防ぐ

    def get_logger(self) -> logging.Logger:
        return self.logger
