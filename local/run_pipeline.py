"""
Run downstream pipeline: calibration -> cost model -> dashboard.

Assumes PDFs and extractions already exist in local/output/.
Use this to iterate on calibration, cost model, or dashboard without
re-generating PDFs or re-running LLM extraction.

Usage:
    python local/run_pipeline.py               # run 03 -> 04 -> 05
    python local/run_pipeline.py --from 04     # start from cost model
    python local/run_pipeline.py --from 05     # dashboard only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    ("03", SCRIPT_DIR / "03_calibration_and_routing.py", "Calibration & Routing"),
    ("04", SCRIPT_DIR / "04_cost_model.py", "Cost Model"),
    ("05", SCRIPT_DIR / "05_portfolio_dashboard.py", "Portfolio Dashboard"),
]


def main():
    parser = argparse.ArgumentParser(description="Run downstream pipeline (no PDFs, no extraction)")
    parser.add_argument(
        "--from", dest="start_from", default="03",
        choices=["03", "04", "05"],
        help="Start from this step (default: 03)"
    )
    args = parser.parse_args()

    # Filter steps
    start_idx = next(i for i, (num, _, _) in enumerate(STEPS) if num == args.start_from)
    steps_to_run = STEPS[start_idx:]

    print(f"Pipeline: {' -> '.join(name for _, _, name in steps_to_run)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    t_start = time.time()

    for step_num, script_path, step_name in steps_to_run:
        print(f"\n--- Step {step_num}: {step_name} ---", flush=True)
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=str(SCRIPT_DIR.parent),
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\nStep {step_num} failed (exit code {result.returncode}). Stopping pipeline.", flush=True)
            sys.exit(result.returncode)

        print(f"--- Step {step_num} complete ({elapsed:.1f}s) ---", flush=True)

    total = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"Pipeline complete in {total:.1f}s", flush=True)
    print(f"Dashboard: {SCRIPT_DIR / 'output' / 'dashboard.html'}", flush=True)


if __name__ == "__main__":
    main()
