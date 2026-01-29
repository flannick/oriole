import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "oriole" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
