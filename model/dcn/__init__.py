from pathlib import Path
import sys

_dcn_dir = str(Path(__file__).resolve().parent)

if _dcn_dir not in sys.path:
    sys.path.insert(0, _dcn_dir)