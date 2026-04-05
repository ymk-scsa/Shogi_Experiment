"""
diagnose_psv.py
PSV (PackedSfenValue) ファイルが正しく読み込めるか、
盤面や指し手が期待通りにパースできるかを確認する診断スクリプト。
"""

import os
import sys
import numpy as np

try:
    import torch
    import cshogi
    from cshogi import Board, PackedSfenValue, move16_from_psv
    print(f"cshogi version: {getattr(cshogi, '__version__', 'unknown')}")
except ImportError as e:
    print(f"Error: 依存ライブラリが不足しています。 ({e})")
    sys.exit(1)

# プロジェクトルートを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Shogi_Experience.GNN_Experiment_20251229.data.past_buffer import PsvDataLoader

def diagnose(file_path):
    if not os.path.exists(file_path):
        print(f"Error: ファイルが見つかりません: {file_path}")
        return

    print(f"--- Diagnosing: {file_path} ---")
    
    # 1. 直接 numpy で構造を確認
    try:
        raw_data = np.fromfile(file_path, dtype=PackedSfenValue, count=10)
        print(f"Successfully read first 10 entries using PackedSfenValue dtype.")
        print(f"Sample 0: score={raw_data[0]['score']}, move={raw_data[0]['move']}, ply={raw_data[0]['gamePly']}")
    except Exception as e:
        print(f"Error reading with numpy: {e}")
        return

    # 2. PsvDataLoader を経由してバッチ取得を確認
    try:
        device = torch.device("cpu")
        loader = PsvDataLoader(file_path, batch_size=4, device=device, shuffle=False)
        print(f"Total entries in file: {len(loader)}")
        
        # 1バッチ取得
        features, moves, values = loader.sample()
        
        print(f"Batch features shape: {features.shape}")
        print(f"Batch moves (labels): {moves}")
        print(f"Batch values (teachers): {values.squeeze()}")
        
        # 3. 盤面表示テスト (最初の1件)
        board = Board()
        board.set_psfen(raw_data[0]['sfen'])
        print("\n--- Board at entry 0 ---")
        print(board)
        m16 = move16_from_psv(int(raw_data[0]['move']))
        print(f"Move from PSV: {board.move_to_usi(m16)}")
        
        print("\nDiagnosis complete. Data looks GOOD.")

    except Exception as e:
        print(f"Error during data loading/parsing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    target = "data/Suisho10Mn_psv.bin"
    diagnose(target)
