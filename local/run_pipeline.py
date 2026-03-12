"""
Run downstream pipeline steps without regenerating PDFs or LLM extraction.

Usage:
    python local/run_pipeline.py                    # Pattern 1: 03 -> 04 -> 05
    python local/run_pipeline.py --pattern 2        # Pattern 2: 07 -> 08
    python local/run_pipeline.py --pattern all      # Both patterns
    python local/run_pipeline.py --from 04          # Pattern 1 from cost model
    python local/run_pipeline.py --from 05          # Pattern 1 dashboard only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS_P1 = [
    ("03", SCRIPT_DIR / "03_calibration_and_routing.py", "Calibration & Routing"),
    ("04", SCRIPT_DIR / "04_cost_model.py", "Cost Model"),
    ("05", SCRIPT_DIR / "05_portfolio_dashboard.py", "Portfolio Dashboard"),
]

STEPS_P2 = [
    ("07", SCRIPT_DIR / "07_generate_valuation_history.py", "Generate Valuation History"),
    ("08", SCRIPT_DIR / "08_valuation_attribution.py", "Valuation Attribution"),
]


def _run_steps(steps, label):
    """Run a list of (step_num, script_path, step_name) tuples sequentially."""
    print(f"\n{label}: {' -> '.join(name for _, _, name in steps)}", flush=True)
    print(f"{'='*60}", flush=True)

    t_start = time.time()

    for step_num, script_path, step_name in steps:
        print(f"\n--- Step {step_num}: {step_name} ---", flush=True)
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=str(SCRIPT_DIR.parent),
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\nStep {step_num} failed (exit code {result.returncode}). Stopping.", flush=True)
            sys.exit(result.returncode)

        print(f"--- Step {step_num} complete ({elapsed:.1f}s) ---", flush=True)

    total = time.time() - t_start
    print(f"\n{label} complete in {total:.1f}s", flush=True)
    return total


def main():
    parser = argparse.ArgumentParser(description="Run pipeline (no PDFs, no extraction)")
    parser.add_argument(
        "--pattern", default="1", choices=["1", "2", "all"],
        help="Which pattern to run (default: 1)"
    )
    parser.add_argument(
        "--from", dest="start_from", default=None,
        choices=["03", "04", "05"],
        help="Start Pattern 1 from this step (default: 03)"
    )
    args = parser.parse_args()

    run_p1 = args.pattern in ("1", "all")
    run_p2 = args.pattern in ("2", "all")

    if run_p1:
        start_from = args.start_from or "03"
        start_idx = next(i for i, (num, _, _) in enumerate(STEPS_P1) if num == start_from)
        _run_steps(STEPS_P1[start_idx:], "Pattern 1")

    if run_p2:
        _run_steps(STEPS_P2, "Pattern 2")

    if run_p1:
        print(f"\nPattern 1 dashboard: {SCRIPT_DIR / 'output' / 'dashboard.html'}", flush=True)
    if run_p2:
        print(f"Pattern 2 dashboard: {SCRIPT_DIR / 'output' / 'model_governance_dashboard.html'}", flush=True)


if __name__ == "__main__":
    main()
