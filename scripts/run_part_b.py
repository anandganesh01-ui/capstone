"""Run Part B customer segments and lapse prediction."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.part_b import main

if __name__ == "__main__":
    main()
