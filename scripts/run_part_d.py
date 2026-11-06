"""Run Part D policy retrieval and the refund agent."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.part_d import main

if __name__ == "__main__":
    main()
