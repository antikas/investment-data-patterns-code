"""
08 — Valuation Attribution & Dashboard (Local)

Reads valuation history produced by script 07, computes quarter-over-quarter
bridges and portfolio-level drift, then generates a self-contained HTML dashboard.

Dashboard sections:
  1. Waterfall bridge  — BYT-2020-I, Q3 2024 (scenario 1: methodology swing)
  2. Assumption drift  — avg buyout EBITDA multiple over 7 quarters (scenario 3)
  3. Cross-GP divergence — Media & Entertainment holdings by GP (scenario 4)
  4. Portfolio decomposition — quarterly NAV change: operational vs assumption vs methodology
  5. Flagged assumptions table — governance log

Usage:
    python local/08_valuation_attribution.py

Requires local/07_generate_valuation_history.py to have been run first.
"""

import base64
import dataclasses
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from shared.fund_definitions import QUARTERS, FUND_DEFINITIONS
from shared.valuation_models import (
    PATTERN_2_OUTPUT_FILES,
    SCENARIO_DIVERGENCE,
    SCENARIO_SWING,
    HoldingRecord,
    AssumptionRecord,
    ValuationRecord,
    BridgeRecord,
    DriftRecord,
    compute_bridges,
    compute_drift,
)

OUTPUT_DIR = PROJECT_ROOT / "local" / "output"
DASHBOARD  = OUTPUT_DIR / "model_governance_dashboard.html"

# ---------------------------------------------------------------------------
# Colours (shared with Pattern 1 dashboard)
# ---------------------------------------------------------------------------
NAVY      = "#1B2A4A"
ACCENT    = "#2E5090"
GREEN     = "#2E7D52"
ORANGE    = "#E07B39"
RED       = "#C0392B"
GREY      = "#666666"
LIGHT_BG  = "#F0F4F8"
WARN      = "#F39C12"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load(key: str) -> list:
    path = OUTPUT_DIR / PATTERN_2_OUTPUT_FILES[key]
    if not path.exists():
        print(f"Error: {path.name} not found. Run local/07_generate_valuation_history.py first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_records(data: list, cls) -> list:
    """Reconstruct dataclass instances from plain dicts."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return [cls(**{k: v for k, v in d.items() if k in fields}) for d in data]


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _chart_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;height:auto;">'


# ---------------------------------------------------------------------------
# Chart 1: Waterfall bridge — BYT-2020-I, Q3 2024
# ---------------------------------------------------------------------------

def chart_waterfall(bridges: list[BridgeRecord], holdings: list[HoldingRecord]) -> str:
    """Fund-level waterfall: opening NAV -> operational -> assumption -> closing NAV."""
    fund_id = SCENARIO_SWING["fund_id"]
    quarter = SCENARIO_SWING["trigger_quarter"]

    fund_bridges = [b for b in bridges if b.fund_id == fund_id and b.quarter == quarter]
    if not fund_bridges:
        return ""

    fund_obj = next((f for f in FUND_DEFINITIONS if f.fund_id == fund_id), None)
    gp_name  = fund_obj.gp_name if fund_obj else fund_id

    opening      = sum(b.opening_fair_value_mm for b in fund_bridges)
    operational  = sum(b.operational_component_mm for b in fund_bridges)
    assumption   = sum(b.assumption_component_mm for b in fund_bridges)
    methodology  = sum(b.methodology_component_mm for b in fund_bridges)
    closing      = sum(b.closing_fair_value_mm for b in fund_bridges)

    prev_q = QUARTERS[QUARTERS.index(quarter) - 1]

    # Build bar list dynamically — methodology bar added only when non-trivial
    labels   = [f"Opening\n({prev_q})", "Operational\nchange", "Assumption\nchange"]
    values   = [opening, operational, assumption]
    bottoms  = [0.0, opening, opening + operational]
    colours  = [NAVY, GREEN if operational >= 0 else RED, ORANGE if assumption >= 0 else RED]
    is_total = [True, False, False]

    if abs(methodology) > 0.1:
        labels.append("Methodology\nchange")
        values.append(methodology)
        bottoms.append(opening + operational + assumption)
        colours.append(RED)
        is_total.append(False)

    labels.append(f"Closing\n({quarter})")
    values.append(closing)
    bottoms.append(0.0)
    colours.append(ACCENT)
    is_total.append(True)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#ffffff")

    for i, (label, val, bot, col, tot) in enumerate(zip(labels, values, bottoms, colours, is_total)):
        if tot:
            ax.bar(i, val, color=col, width=0.55, zorder=3)
            ax.text(i, val + opening * 0.01, f"${val:,.0f}M", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=NAVY)
        else:
            ax.bar(i, abs(val), bottom=bot if val >= 0 else bot + val,
                   color=col, width=0.55, zorder=3)
            sign = "+" if val >= 0 else ""
            ax.text(i, (bot + max(val, 0)) + opening * 0.01,
                    f"{sign}${val:,.0f}M", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=col)

    # Connector lines: draw into every delta bar at its starting level (bottoms[i+1])
    for i in range(len(labels) - 1):
        if not is_total[i + 1]:
            y = bottoms[i + 1]
            ax.plot([i + 0.28, i + 0.72], [y, y], color=GREY, lw=0.8, ls="--")

    assumption_pct = abs(assumption) / abs(closing - opening) * 100 if (closing - opening) != 0 else 0
    ax.set_title(
        f"{gp_name} ({fund_id}) — NAV Bridge {quarter}\n"
        f"{assumption_pct:.0f}% of the change is assumption-driven (post-ECB rate cut repricing)",
        fontsize=11, fontweight="bold", color=NAVY, pad=12,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, color=NAVY)
    ax.set_ylabel("Fair Value ($M)", fontsize=9, color=GREY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.tick_params(axis="y", labelcolor=GREY, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#e0e0e0", lw=0.5, zorder=0)

    # Flagged badge
    ax.text(0.98, 0.97, "⚠ FLAGGED — board review required",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=WARN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9E6", edgecolor=WARN, lw=1))

    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Chart 2: Buyout EBITDA multiple drift over 7 quarters
# ---------------------------------------------------------------------------

def chart_drift(drift: list[DriftRecord]) -> str:
    quarters = [d.quarter for d in drift]
    avg_mult = [d.avg_buyout_ebitda_multiple for d in drift]
    med_mult = [d.median_buyout_ebitda_multiple for d in drift]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#ffffff")

    x = range(len(quarters))
    ax.plot(x, avg_mult, color=ACCENT,  lw=2,   marker="o", ms=6, label="Average multiple",  zorder=3)
    ax.plot(x, med_mult, color=ORANGE,  lw=1.5, marker="s", ms=5, ls="--", label="Median multiple", zorder=3)

    # Shade the area between Q1 2024 baseline and the drift line
    baseline = avg_mult[0]
    ax.fill_between(x, baseline, avg_mult,
                    where=[m >= baseline for m in avg_mult],
                    alpha=0.12, color=ORANGE, label="Assumption expansion vs baseline")

    ax.axhline(baseline, color=GREY, lw=0.8, ls=":", zorder=2)
    ax.text(len(quarters) - 0.6, baseline + 0.05, f"Q1 2024 baseline: {baseline:.2f}x",
            fontsize=7.5, color=GREY, va="bottom")

    # Annotate scenario 1 quarter
    swing_idx = quarters.index(SCENARIO_SWING["trigger_quarter"])
    ax.annotate(
        f"Scenario 1 spike\n({SCENARIO_SWING['fund_id']})",
        xy=(swing_idx, avg_mult[swing_idx]),
        xytext=(swing_idx + 0.4, avg_mult[swing_idx] + 0.25),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1),
        fontsize=7.5, color=RED,
    )

    drift_total = avg_mult[-1] - avg_mult[0]
    ax.set_title(
        f"Buyout Portfolio — EBITDA Multiple Drift: {avg_mult[0]:.2f}x -> {avg_mult[-1]:.2f}x "
        f"(+{drift_total:.2f}x over 7 quarters)\n"
        "Systematic assumption expansion tracking public-market PE repricing",
        fontsize=10, fontweight="bold", color=NAVY, pad=10,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(quarters, rotation=30, ha="right", fontsize=8, color=NAVY)
    ax.set_ylabel("Avg EBITDA Multiple (x)", fontsize=9, color=GREY)
    ax.tick_params(axis="y", labelcolor=GREY, labelsize=8)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#e0e0e0", lw=0.5)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Chart 3: Cross-GP divergence — Media & Entertainment
# ---------------------------------------------------------------------------

def chart_cross_gp(assumptions: list[AssumptionRecord],
                   holdings: list[HoldingRecord]) -> str:
    """Track EBITDA multiples for Media & Entertainment holdings across both GP funds."""
    sector   = SCENARIO_DIVERGENCE["sector"]
    fund_a   = SCENARIO_DIVERGENCE["conservative_fund"]
    fund_b   = SCENARIO_DIVERGENCE["optimistic_fund"]

    # Build holding_id -> (fund_id, company_name) map for target sector
    target_holdings = {h.holding_id: h for h in holdings
                       if h.sector == sector and h.fund_id in (fund_a, fund_b)}

    if not target_holdings:
        return ""

    # Per-holding EBITDA multiples over time
    fund_a_series: dict[str, list] = defaultdict(list)
    fund_b_series: dict[str, list] = defaultdict(list)

    for q in QUARTERS:
        for hid, h in target_holdings.items():
            rec = next((a for a in assumptions if a.holding_id == hid and a.quarter == q), None)
            if rec and rec.ebitda_multiple is not None:
                if h.fund_id == fund_a:
                    fund_a_series[h.company_name].append(rec.ebitda_multiple)
                else:
                    fund_b_series[h.company_name].append(rec.ebitda_multiple)

    if not fund_a_series and not fund_b_series:
        return ""

    fund_a_obj = next((f for f in FUND_DEFINITIONS if f.fund_id == fund_a), None)
    fund_b_obj = next((f for f in FUND_DEFINITIONS if f.fund_id == fund_b), None)
    gp_a = fund_a_obj.gp_name if fund_a_obj else fund_a
    gp_b = fund_b_obj.gp_name if fund_b_obj else fund_b

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#ffffff")

    x = list(range(len(QUARTERS)))

    for company, series in fund_a_series.items():
        if len(series) == len(QUARTERS):
            ax.plot(x, series, color=ACCENT, lw=2, marker="o", ms=5, label=f"{gp_a}: {company}")

    for company, series in fund_b_series.items():
        if len(series) == len(QUARTERS):
            ax.plot(x, series, color=ORANGE, lw=2, marker="s", ms=5, label=f"{gp_b}: {company}")

    # Avg lines per GP — only full-length series (matches what gets plotted)
    full_a = [s for s in fund_a_series.values() if len(s) == len(QUARTERS)]
    if full_a:
        avg_a = [sum(q_vals[i] for q_vals in full_a) / len(full_a) for i in range(len(QUARTERS))]
        ax.plot(x, avg_a, color=ACCENT, lw=3, ls="--", alpha=0.5)

    full_b = [s for s in fund_b_series.values() if len(s) == len(QUARTERS)]
    if full_b:
        avg_b = [sum(q_vals[i] for q_vals in full_b) / len(full_b) for i in range(len(QUARTERS))]
        ax.plot(x, avg_b, color=ORANGE, lw=3, ls="--", alpha=0.5)

    # Annotate the gap in Q3 2025 (use same filtered series as avg lines)
    if full_a and full_b:
        last_a = [v[-1] for v in full_a]
        last_b = [v[-1] for v in full_b]
        avg_last_a = sum(last_a) / len(last_a)
        avg_last_b = sum(last_b) / len(last_b)
        gap = avg_last_b - avg_last_a
        ax.annotate(
            f"delta: {gap:+.1f}x\nsame sector,\ndifferent peers",
            xy=(len(QUARTERS) - 1, (avg_last_a + avg_last_b) / 2),
            xytext=(len(QUARTERS) - 1.8, (avg_last_a + avg_last_b) / 2 + 0.4),
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8),
            fontsize=7.5, color=RED, fontweight="bold",
        )

    ax.set_title(
        f"Cross-GP Divergence — {sector}\n"
        f"{gp_a} (conservative peers) vs {gp_b} (growth peers): same sector, different implied multiples",
        fontsize=10, fontweight="bold", color=NAVY, pad=10,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(QUARTERS, rotation=30, ha="right", fontsize=8, color=NAVY)
    ax.set_ylabel("EBITDA Multiple (x)", fontsize=9, color=GREY)
    ax.tick_params(axis="y", labelcolor=GREY, labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#e0e0e0", lw=0.5)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Chart 4: Portfolio decomposition — stacked bar by quarter
# ---------------------------------------------------------------------------

def chart_decomposition(drift: list[DriftRecord]) -> str:
    """Stacked bar: for each quarter with bridges, show operational vs assumption breakdown."""
    # Bridges exist from Q2 2024 onwards (index 1+)
    bridge_quarters = QUARTERS[1:]
    drift_map       = {d.quarter: d for d in drift}

    op_vals   = [drift_map[q].operational_change_mm   for q in bridge_quarters]
    asmp_vals = [drift_map[q].assumption_change_mm    for q in bridge_quarters]
    meth_vals = [drift_map[q].methodology_change_mm   for q in bridge_quarters]

    x     = np.arange(len(bridge_quarters))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#ffffff")

    # Separate positive and negative for stacked bars
    def _pos(v): return max(v, 0)
    def _neg(v): return min(v, 0)

    # Positive stack
    b1 = ax.bar(x, [_pos(v) for v in op_vals],   width, color=GREEN,  label="Operational",  zorder=3)
    b2 = ax.bar(x, [_pos(v) for v in asmp_vals],  width,
                bottom=[_pos(o) for o in op_vals],
                color=ORANGE, label="Assumption", zorder=3)
    b3 = ax.bar(x, [_pos(v) for v in meth_vals],  width,
                bottom=[_pos(o) + _pos(a) for o, a in zip(op_vals, asmp_vals)],
                color=RED, label="Methodology", zorder=3)

    # Negative stack (lighter alpha distinguishes negative components)
    ax.bar(x, [_neg(v) for v in op_vals],   width, color=GREEN,  alpha=0.6, zorder=3)
    ax.bar(x, [_neg(v) for v in asmp_vals],  width,
           bottom=[_neg(o) for o in op_vals],
           color=ORANGE, alpha=0.6, zorder=3)
    ax.bar(x, [_neg(v) for v in meth_vals],  width,
           bottom=[_neg(o) + _neg(a) for o, a in zip(op_vals, asmp_vals)],
           color=RED, alpha=0.6, zorder=3)

    has_negatives = any(v < 0 for v in op_vals + asmp_vals + meth_vals)
    if has_negatives:
        ax.text(0.98, 0.02, "Lighter bars = negative components",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=GREY, style="italic")

    ax.axhline(0, color=NAVY, lw=0.8)

    # Annotate Q3 2024 scenario 1 — big assumption bar
    swing_idx = bridge_quarters.index(SCENARIO_SWING["trigger_quarter"])
    total_assumption_swing = asmp_vals[swing_idx]
    ax.annotate(
        f"Scenario 1: +${total_assumption_swing:,.0f}M\nassumption-driven",
        xy=(swing_idx, _pos(op_vals[swing_idx]) + _pos(total_assumption_swing)),
        xytext=(swing_idx + 0.6, _pos(op_vals[swing_idx]) + _pos(total_assumption_swing) + 30),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1),
        fontsize=7.5, color=ORANGE,
    )

    ax.set_title(
        "Portfolio NAV Change by Quarter — Operational vs Assumption vs Methodology\n"
        "Q3 2024 assumption spike = BYT-2020-I repricing. Systematic drift visible across all quarters.",
        fontsize=10, fontweight="bold", color=NAVY, pad=10,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(bridge_quarters, rotation=30, ha="right", fontsize=8, color=NAVY)
    ax.set_ylabel("NAV Change ($M)", fontsize=9, color=GREY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}M"))
    ax.tick_params(axis="y", labelcolor=GREY, labelsize=8)
    ax.legend(fontsize=8.5, framealpha=0.8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#e0e0e0", lw=0.5, zorder=0)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _flagged_table(assumptions: list[AssumptionRecord],
                   holdings: list[HoldingRecord]) -> str:
    hold_map = {h.holding_id: h for h in holdings}
    flagged  = sorted(
        [a for a in assumptions if a.approval_status == "flagged"],
        key=lambda a: (a.fund_id, a.quarter),
    )
    if not flagged:
        return "<p>No flagged assumptions.</p>"

    source_labels = {"disclosed": "Disclosed", "back_calculated": "Back-calc", "extracted": "Extracted"}

    rows = ""
    for a in flagged:
        h     = hold_map.get(a.holding_id)
        co    = h.company_name if h else a.holding_id
        mult  = a.ebitda_multiple or a.revenue_multiple or "—"
        note  = (a.notes or "")[:120]
        src   = source_labels.get(a.assumption_source, a.assumption_source)
        rows += (
            f"<tr>"
            f"<td>{a.fund_id}</td><td>{a.quarter}</td><td>{co}</td>"
            f"<td>{a.method}</td>"
            f"<td style='font-weight:bold;color:{RED}'>{mult if isinstance(mult, str) else f'{mult:.2f}x'}</td>"
            f"<td>{src}</td>"
            f"<td><span style='background:#FFF9E6;padding:2px 6px;border-radius:3px;color:{WARN}'>⚠ flagged</span></td>"
            f"<td style='font-size:0.85em;color:{GREY}'>{note}</td>"
            f"</tr>\n"
        )

    return f"""
<table style="width:100%;border-collapse:collapse;font-size:0.9em;">
  <thead>
    <tr style="background:{NAVY};color:white;">
      <th style="padding:8px 10px;text-align:left;">Fund</th>
      <th>Quarter</th>
      <th>Company</th>
      <th>Method</th>
      <th>Multiple</th>
      <th>Source</th>
      <th>Status</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>"""


def _kpi_box(label: str, value: str, note: str = "", warn: bool = False) -> str:
    colour = WARN if warn else ACCENT
    return f"""
<div style="background:white;border-left:4px solid {colour};padding:14px 18px;
            border-radius:0 6px 6px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);">
  <div style="font-size:1.6em;font-weight:700;color:{colour}">{value}</div>
  <div style="font-size:0.9em;color:{NAVY};font-weight:600;margin-top:2px">{label}</div>
  {f'<div style="font-size:0.78em;color:{GREY};margin-top:4px">{note}</div>' if note else ''}
</div>"""


def build_html(
    holdings:    list[HoldingRecord],
    assumptions: list[AssumptionRecord],
    valuations:  list[ValuationRecord],
    bridges:     list[BridgeRecord],
    drift:       list[DriftRecord],
) -> str:
    # KPIs
    total_holdings  = len(holdings)
    total_fv        = drift[-1].total_portfolio_nav_mm
    flagged_count   = sum(1 for a in assumptions if a.approval_status == "flagged")
    changed_count   = sum(1 for a in assumptions if a.assumption_changed)
    cum_op          = sum(d.operational_change_mm  for d in drift[1:])
    cum_assump      = sum(d.assumption_change_mm   for d in drift[1:])
    cum_total       = cum_op + cum_assump
    assump_pct      = abs(cum_assump) / abs(cum_total) * 100 if cum_total else 0

    # Assumption source coverage (latest quarter)
    latest = drift[-1]
    src_total = latest.nav_disclosed_mm + latest.nav_back_calculated_mm + latest.nav_extracted_mm
    if src_total > 0:
        pct_disclosed = latest.nav_disclosed_mm / src_total * 100
        pct_back_calc = latest.nav_back_calculated_mm / src_total * 100
        pct_extracted = latest.nav_extracted_mm / src_total * 100
    else:
        pct_disclosed = pct_back_calc = pct_extracted = 0.0

    # Charts
    wf  = chart_waterfall(bridges, holdings)
    dr  = chart_drift(drift)
    xgp = chart_cross_gp(assumptions, holdings)
    dc  = chart_decomposition(drift)

    wf_tag  = _chart_tag(wf,  "Waterfall bridge")  if wf  else ""
    dr_tag  = _chart_tag(dr,  "Assumption drift")  if dr  else ""
    xgp_tag = _chart_tag(xgp, "Cross-GP divergence") if xgp else ""
    dc_tag  = _chart_tag(dc,  "Portfolio decomposition") if dc else ""

    flag_tbl = _flagged_table(assumptions, holdings)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pattern 2: Mark-to-Model Governance Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: {LIGHT_BG}; color: {NAVY}; font-size: 14px; }}
  .header {{ background: {NAVY}; color: white; padding: 24px 40px; }}
  .header h1 {{ font-size: 1.5em; font-weight: 700; margin-bottom: 4px; }}
  .header p  {{ font-size: 0.9em; opacity: 0.75; }}
  .content {{ max-width: 1200px; margin: 0 auto; padding: 28px 24px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
               gap: 14px; margin-bottom: 32px; }}
  .section {{ background: white; border-radius: 8px; padding: 24px;
              box-shadow: 0 1px 6px rgba(0,0,0,.07); margin-bottom: 28px; }}
  .section h2 {{ font-size: 1.05em; font-weight: 700; color: {NAVY};
                border-bottom: 2px solid {LIGHT_BG}; padding-bottom: 10px; margin-bottom: 16px; }}
  .scenario-badge {{ display:inline-block; background:{LIGHT_BG}; color:{ACCENT};
                     font-size:0.75em; font-weight:700; padding:2px 8px; border-radius:12px;
                     margin-left:8px; vertical-align:middle; }}
  .insight {{ background:{LIGHT_BG}; border-left:3px solid {ACCENT}; padding:10px 14px;
              border-radius:0 6px 6px 0; font-size:0.88em; color:{NAVY}; margin-top:14px; }}
  footer {{ text-align:center; padding:18px; font-size:0.78em; color:{GREY}; }}
</style>
</head>
<body>

<div class="header">
  <h1>Pattern 2: Mark-to-Model Governance</h1>
  <p>Investment Data Patterns Demo &nbsp;·&nbsp; {total_holdings} holdings across 4 strategies
     &nbsp;·&nbsp; 7 quarters (Q1 2024 – Q3 2025) &nbsp;·&nbsp; Generated {generated}</p>
</div>

<div class="content">

  <!-- KPIs -->
  <div class="kpi-grid">
    {_kpi_box("Holdings", str(total_holdings), "Buyout · Growth Equity · Infra · Real Estate")}
    {_kpi_box("Portfolio NAV (Q3 2025)", f"${total_fv:,.0f}M", "Mark-to-model fair values")}
    {_kpi_box("Assumption-driven NAV change", f"{assump_pct:.0f}%",
              f"of cumulative change over 7 quarters", warn=assump_pct > 30)}
    {_kpi_box("Assumption changes logged", str(changed_count), "Tracked per holding per quarter")}
    {_kpi_box("Flagged for review", str(flagged_count),
              "Pending board approval", warn=flagged_count > 0)}
    {_kpi_box("Assumption Data Sources",
              f"{pct_disclosed:.0f}% / {pct_back_calc:.0f}% / {pct_extracted:.0f}%",
              "Disclosed / Back-calculated / Extracted (by NAV)")}
  </div>

  <!-- Chart 1: Waterfall -->
  <div class="section">
    <h2>Scenario 1: Methodology-Driven Swing
      <span class="scenario-badge">BYT-2020-I · Q3 2024</span>
    </h2>
    {wf_tag or '<p style="color:#999">No bridge data for this scenario.</p>'}
    <div class="insight">
      The GP adjusted EBITDA multiples sharply upward following ECB rate cuts. Operations were flat.
      The bridge shows that virtually all of the NAV movement came from assumption change —
      not from anything the underlying companies did. Without decomposition, this looks like
      outperformance. With it, the cause is transparent. This assumption set was flagged for board review.
    </div>
  </div>

  <!-- Chart 2: Drift timeline -->
  <div class="section">
    <h2>Scenario 3: Systematic Assumption Drift
      <span class="scenario-badge">All Buyout Funds · 7 Quarters</span>
    </h2>
    {dr_tag or '<p style="color:#999">No drift data.</p>'}
    <div class="insight">
      Across all buyout funds, the average EBITDA multiple drifted upward quarter by quarter —
      tracking public-market PE repricing during the rate-cut cycle.
      No single quarter's change looks alarming. The cumulative drift is significant.
      Portfolio-level drift detection surfaces this pattern that per-fund review misses.
    </div>
  </div>

  <!-- Chart 3: Cross-GP -->
  <div class="section">
    <h2>Scenario 4: Cross-GP Divergence
      <span class="scenario-badge">Media &amp; Entertainment Sector</span>
    </h2>
    {xgp_tag or '<p style="color:#999">No cross-GP data for this sector.</p>'}
    <div class="insight">
      Two GPs hold comparable Media &amp; Entertainment assets. The same sector.
      Similar underlying businesses. But one GP selected a conservative comparable set
      (traditional media peers); the other selected a digital-premium comparable set.
      The result: a persistent multi-turn multiple gap. Both assumptions are defensible in isolation.
      The divergence is only visible when you compare across GPs.
    </div>
  </div>

  <!-- Chart 4: Decomposition -->
  <div class="section">
    <h2>Portfolio NAV Decomposition — 6 Quarters of Bridges</h2>
    {dc_tag or '<p style="color:#999">No decomposition data.</p>'}
    <div class="insight">
      Each quarter's NAV change is decomposed into what the businesses earned (operational)
      and what the valuation assumptions contributed (assumption). A well-governed portfolio
      should show assumption changes that are explainable and approved — not silent drift.
      The Q3 2024 spike is scenario 1; the steady assumption component in other quarters is scenario 3.
    </div>
  </div>

  <!-- Table: flagged assumptions -->
  <div class="section">
    <h2>Governance Log — Flagged Assumptions</h2>
    {flag_tbl}
    <div class="insight">
      The governance log captures every assumption that triggered a review flag.
      In a real system this would include the approver, approval date, and supporting
      memo reference — linking every fair value back to an auditable decision.
    </div>
  </div>

</div>

<footer>Pattern 2: Mark-to-Model Governance · Investment Data Patterns Demo · Synthetic data only</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Pattern 2: Computing attribution and building dashboard\n")

    # Load
    holdings    = _to_records(_load("holdings"),    HoldingRecord)
    assumptions = _to_records(_load("assumptions"), AssumptionRecord)
    valuations  = _to_records(_load("valuations"),  ValuationRecord)

    print(f"  Loaded {len(holdings)} holdings, {len(assumptions)} assumptions, "
          f"{len(valuations)} valuations")

    # Compute
    bridges = compute_bridges(holdings, assumptions, valuations)
    drift   = compute_drift(assumptions, valuations, bridges)

    print(f"  Computed {len(bridges)} bridges across {len(QUARTERS)-1} quarter transitions")

    # Save attribution outputs
    def _save(records, key):
        path = OUTPUT_DIR / PATTERN_2_OUTPUT_FILES[key]
        data = [dataclasses.asdict(r) for r in records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  {len(data):>4} records  ->  {path.name}")

    _save(bridges, "bridges")
    _save(drift,   "drift")

    # Print scenario 2 bridge as sanity check
    from shared.valuation_models import SCENARIO_HIDDEN, _holding_id
    hid_helios = _holding_id(SCENARIO_HIDDEN["fund_id"], SCENARIO_HIDDEN["company_name"])
    helios_bridge = next((b for b in bridges
                          if b.holding_id == hid_helios
                          and b.quarter == SCENARIO_HIDDEN["trigger_quarter"]), None)
    if helios_bridge:
        print(f"\nScenario 2: {SCENARIO_HIDDEN['company_name']} bridge ({helios_bridge.quarter}):")
        print(f"  Total change:        ${helios_bridge.total_change_mm:+.1f}M")
        print(f"  Operational:         ${helios_bridge.operational_component_mm:+.1f}M")
        print(f"  Assumption:          ${helios_bridge.assumption_component_mm:+.1f}M")
        print(f"  {helios_bridge.assumption_change_description}")

    # Build dashboard
    print("\nGenerating dashboard...")
    html = build_html(holdings, assumptions, valuations, bridges, drift)
    with open(DASHBOARD, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDashboard: {DASHBOARD}")
    print("Open in a browser to review.")


if __name__ == "__main__":
    main()
