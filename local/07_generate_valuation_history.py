"""
07 — Generate Valuation History (Local)

Generates synthetic mark-to-model valuation data for all mark-to-model strategies
(Buyout, Growth Equity, Infrastructure, Real Estate): 19 funds, ~60 holdings,
7 quarters (Q1 2024 – Q3 2025).

Four governance scenarios are embedded in the synthetic data:
  1. Methodology swing:     BYT-2020-I, Q3 2024 — assumption-driven NAV move, flagged
  2. Hidden outperformance: GEQ-2021-I/Helios, Q2 2025 — strong ops masked by tightening
  3. Systematic drift:      all buyout funds +0.10x/quarter (tracks public PE repricing)
  4. Cross-GP divergence:   BYT-2019-I vs BYT-2022-I, same sector, different comp sets

Outputs (to local/output/):
  model_holdings.json
  model_assumptions.json
  model_valuations.json
  model_data_access.json

model_bridges.json and model_drift.json are produced by script 08.

Usage:
    python local/07_generate_valuation_history.py

No external dependencies. No LLM calls.
"""

import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.fund_definitions import QUARTERS
from shared.valuation_models import (
    DATA_ACCESS_RECORDS,
    MARK_TO_MODEL_STRATEGIES,
    PATTERN_2_OUTPUT_FILES,
    SCENARIO_DIVERGENCE,
    SCENARIO_HIDDEN,
    SCENARIO_SWING,
    generate_holdings,
    generate_valuation_history,
)

OUTPUT_DIR = PROJECT_ROOT / "local" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_records(records: list, key: str) -> None:
    path = OUTPUT_DIR / PATTERN_2_OUTPUT_FILES[key]
    data = [dataclasses.asdict(r) for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  {len(data):>4} records  ->  {path.name}")


# ---------------------------------------------------------------------------
# Scenario summaries
# ---------------------------------------------------------------------------

def _print_scenario_1(valuations, assumptions):
    """Scenario 1: show assumption jump for BYT-2020-I in Q3 2024."""
    fund_id = SCENARIO_SWING["fund_id"]
    trigger = SCENARIO_SWING["trigger_quarter"]
    prev_q  = QUARTERS[QUARTERS.index(trigger) - 1]

    fund_assumps = [a for a in assumptions if a.fund_id == fund_id]
    prev_multiples = [a.ebitda_multiple for a in fund_assumps
                      if a.quarter == prev_q and a.ebitda_multiple is not None]
    curr_multiples = [a.ebitda_multiple for a in fund_assumps
                      if a.quarter == trigger and a.ebitda_multiple is not None]
    flagged        = [a for a in fund_assumps
                      if a.quarter == trigger and a.approval_status == "flagged"]

    if prev_multiples and curr_multiples:
        avg_prev = sum(prev_multiples) / len(prev_multiples)
        avg_curr = sum(curr_multiples) / len(curr_multiples)
        print(f"\nScenario 1:{SCENARIO_SWING['fund_id']} assumption swing ({prev_q} ->{trigger}):")
        print(f"  Avg EBITDA multiple: {avg_prev:.2f}x ->{avg_curr:.2f}x  (delta:{avg_curr - avg_prev:+.2f}x)")
        print(f"  Flagged assumptions: {len(flagged)}")


def _print_scenario_2(valuations, assumptions):
    """Scenario 2: show hidden outperformance for Helios Data Systems."""
    fund_id  = SCENARIO_HIDDEN["fund_id"]
    company  = SCENARIO_HIDDEN["company_name"]
    trigger  = SCENARIO_HIDDEN["trigger_quarter"]
    prev_q   = QUARTERS[QUARTERS.index(trigger) - 1]

    # Find the holding
    from shared.valuation_models import _holding_id
    hid = _holding_id(fund_id, company)
    prev_v = next((v for v in valuations if v.holding_id == hid and v.quarter == prev_q), None)
    curr_v = next((v for v in valuations if v.holding_id == hid and v.quarter == trigger), None)
    prev_a = next((a for a in assumptions if a.holding_id == hid and a.quarter == prev_q), None)
    curr_a = next((a for a in assumptions if a.holding_id == hid and a.quarter == trigger), None)

    if prev_v and curr_v and prev_a and curr_a:
        metric_growth = (curr_v.underlying_metric_mm / prev_v.underlying_metric_mm - 1) * 100
        nav_change    = curr_v.fair_value_mm - prev_v.fair_value_mm
        mult_prev     = prev_a.revenue_multiple or 0
        mult_curr     = curr_a.revenue_multiple or 0
        print(f"\nScenario 2:{company} ({prev_q} ->{trigger}):")
        print(f"  Revenue:  {prev_v.underlying_metric_mm:.1f}M ->{curr_v.underlying_metric_mm:.1f}M  (+{metric_growth:.0f}%)")
        print(f"  Multiple: {mult_prev:.2f}x ->{mult_curr:.2f}x  (delta:{mult_curr - mult_prev:+.2f}x)")
        print(f"  Fair value change: {nav_change:+.1f}M  (ops masked by multiple compression)")


def _print_scenario_4(assumptions, holdings):
    """Scenario 4: show cross-GP divergence for Media & Entertainment."""
    sector = SCENARIO_DIVERGENCE["sector"]
    fund_a = SCENARIO_DIVERGENCE["conservative_fund"]
    fund_b = SCENARIO_DIVERGENCE["optimistic_fund"]
    q_last = QUARTERS[-1]

    # Build set of holding_ids in the target sector for each fund
    hids_a = {h.holding_id for h in holdings if h.fund_id == fund_a and h.sector == sector}
    hids_b = {h.holding_id for h in holdings if h.fund_id == fund_b and h.sector == sector}

    mults_a = [a.ebitda_multiple for a in assumptions
               if a.holding_id in hids_a and a.quarter == q_last and a.ebitda_multiple is not None]
    mults_b = [a.ebitda_multiple for a in assumptions
               if a.holding_id in hids_b and a.quarter == q_last and a.ebitda_multiple is not None]

    if mults_a and mults_b:
        print(f"\nScenario 4 - Cross-GP divergence ({sector}) in {q_last}:")
        print(f"  {fund_a} (conservative peers): avg {sum(mults_a)/len(mults_a):.2f}x EBITDA")
        print(f"  {fund_b} (growth peers):        avg {sum(mults_b)/len(mults_b):.2f}x EBITDA")
    else:
        print(f"\nScenario 4 - No {sector} holdings found in {fund_a} or {fund_b}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Pattern 2: Generating valuation history")
    print(f"Strategies: {', '.join(sorted(MARK_TO_MODEL_STRATEGIES))}\n")

    holdings = generate_holdings()

    strategy_counts: dict = {}
    for h in holdings:
        strategy_counts[h.strategy] = strategy_counts.get(h.strategy, 0) + 1
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  {strategy}: {count} holdings")
    print(f"  Total: {len(holdings)} holdings x {len(QUARTERS)} quarters\n")

    assumptions, valuations = generate_valuation_history(holdings)

    print(f"  {len(assumptions)} assumption records")
    print(f"  {len(valuations)} valuation records\n")

    print("Writing output files:")
    save_records(holdings,            "holdings")
    save_records(assumptions,         "assumptions")
    save_records(valuations,          "valuations")
    save_records(DATA_ACCESS_RECORDS, "data_access")

    # Scenario summaries for quick sanity check
    _print_scenario_1(valuations, assumptions)
    _print_scenario_2(valuations, assumptions)
    _print_scenario_4(assumptions, holdings)

    # Flagged assumptions
    flagged = [a for a in assumptions if a.approval_status == "flagged"]
    print(f"\nFlagged assumptions requiring review: {len(flagged)}")
    for a in flagged[:6]:
        note = (a.notes or "")[:70]
        print(f"  {a.fund_id} | {a.quarter} | {a.assumption_id.split('_')[-3] or ''} | {note}")

    print(f"\nDone. Run local/08_valuation_attribution.py to compute bridges and build the dashboard.")


if __name__ == "__main__":
    main()
