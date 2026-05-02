"""USI エントリ（例: python main.py < script.txt）。"""

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from player.npls_player import NPLSPlayer

if __name__ == "__main__":
    try:
        NPLSPlayer().run()
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
