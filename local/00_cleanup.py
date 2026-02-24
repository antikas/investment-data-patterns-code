"""Delete local/output/ directory tree. Safe to run at any point."""

import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
    print(f"Deleted: {OUTPUT_DIR}")
else:
    print("Nothing to clean")
