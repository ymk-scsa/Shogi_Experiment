import os


def ensure_directory_exists(path: str):
    """
    指定したパスのディレクトリ部分が存在しない場合に作成する。

    Args:
        path (str): 確認・作成するディレクトリまたはファイルのパス
    """
    directory_path = os.path.dirname(path) if os.path.splitext(path)[1] else path

    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)
