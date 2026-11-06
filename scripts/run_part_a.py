"""Run the Part A audit, cleaning, validation, SQL, and exploratory analysis."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.part_a import main

if __name__ == "__main__":
    main()
