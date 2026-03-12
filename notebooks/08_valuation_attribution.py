# Databricks notebook source
"""
08 - Valuation Attribution & Dashboard (Mark-to-Model Governance)

Reads valuation history from Delta (written by notebook 07), computes
quarter-over-quarter bridges and portfolio-level drift, then renders
charts inline using Databricks display().

Charts:
  1. Waterfall bridge  - BYT-2020-I, Q3 2024 (scenario 1: methodology swing)
  2. Assumption drift  - avg buyout EBITDA multiple over 7 quarters (scenario 3)
  3. Cross-GP divergence - Media & Entertainment holdings by GP (scenario 4)
  4. Portfolio decomposition - quarterly NAV change: operational vs assumption vs methodology

Local equivalent: local/08_valuation_attribution.py
No LLM calls.
"""

# COMMAND ----------
# MAGIC %md
# MAGIC # 08 - Valuation Attribution & Governance Dashboard
# MAGIC
# MAGIC Computes **bridge decomposition** (operational vs assumption vs methodology) and
# MAGIC **portfolio-level drift** from the valuation history in notebook 07.
# MAGIC
# MAGIC **Output tables:**
# MAGIC - `model_bridges` - quarterly NAV attribution per holding
# MAGIC - `model_drift`   - portfolio-level assumption trends per quarter
# MAGIC
# MAGIC **Governance scenarios visualised:**
# MAGIC | # | Scenario | Fund | Quarter |
# MAGIC |---|----------|------|---------|
# MAGIC | 1 | Methodology-driven swing | BYT-2020-I | Q3 2024 |
# MAGIC | 2 | Hidden outperformance (bridge only) | GEQ-2021-I / Helios | Q2 2025 |
# MAGIC | 3 | Systematic drift | All buyout funds | Q1-Q3 2025 |
# MAGIC | 4 | Cross-GP divergence | BYT-2019-I vs BYT-2022-I | Media & Ent. |

# COMMAND ----------

# Create widgets with defaults (idempotent - preserves user's current value)
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schema", "extraction_demo", "Schema Name")
catalog = dbutils.widgets.get("catalog")
schema  = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Target: {catalog}.{schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Import Shared Modules

# COMMAND ----------

import dataclasses
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

notebook_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
sys.path.insert(0, project_root)

from shared.fund_definitions import FUND_DEFINITIONS, QUARTERS
from shared.valuation_models import (
    SCENARIO_DIVERGENCE,
    SCENARIO_HIDDEN,
    SCENARIO_SWING,
    HoldingRecord,
    AssumptionRecord,
    ValuationRecord,
    BridgeRecord,
    DriftRecord,
    _holding_id,
    compute_bridges,
    compute_drift,
)

print(f"Quarters: {len(QUARTERS)} ({QUARTERS[0]} to {QUARTERS[-1]})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Colour Palette

# COMMAND ----------

NAVY     = "#1B2A4A"
ACCENT   = "#2E5090"
GREEN    = "#2E7D52"
ORANGE   = "#E07B39"
RED      = "#C0392B"
GREY     = "#666666"
LIGHT_BG = "#F0F4F8"
WARN     = "#F39C12"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Data from Delta Tables

# COMMAND ----------

def _load_from_delta(table_name: str, cls):
    """Read Delta table and reconstruct dataclass instances. Handles NaN -> None for optional fields."""
    pdf = spark.table(f"{catalog}.{schema}.{table_name}").toPandas()
    pdf = pdf.where(pdf.notna(), other=None)   # NaN -> None for Optional fields
    fields = {f.name for f in dataclasses.fields(cls)}
    return [cls(**{k: v for k, v in row.items() if k in fields})
            for row in pdf.to_dict("records")]

holdings    = _load_from_delta("model_holdings",    HoldingRecord)
assumptions = _load_from_delta("model_assumptions", AssumptionRecord)
valuations  = _load_from_delta("model_valuations",  ValuationRecord)

print(f"Loaded: {len(holdings)} holdings, {len(assumptions)} assumptions, {len(valuations)} valuations")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compute Attribution Bridges

# COMMAND ----------

bridges = compute_bridges(holdings, assumptions, valuations)

print(f"Computed {len(bridges)} bridges across {len(QUARTERS)-1} quarter transitions")

# Scenario 2 sanity check
hid_helios    = _holding_id(SCENARIO_HIDDEN["fund_id"], SCENARIO_HIDDEN["company_name"])
helios_bridge = next((b for b in bridges
                      if b.holding_id == hid_helios
                      and b.quarter == SCENARIO_HIDDEN["trigger_quarter"]), None)
if helios_bridge:
    print(f"\nScenario 2: {SCENARIO_HIDDEN['company_name']} bridge ({helios_bridge.quarter}):")
    print(f"  Total:       ${helios_bridge.total_change_mm:+.1f}M")
    print(f"  Operational: ${helios_bridge.operational_component_mm:+.1f}M")
    print(f"  Assumption:  ${helios_bridge.assumption_component_mm:+.1f}M  <-- ops masked by multiple compression")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compute Portfolio Drift

# COMMAND ----------

drift = compute_drift(assumptions, valuations, bridges)
print(f"Drift records: {len(drift)} quarters")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write to Delta Tables

# COMMAND ----------

def _save_to_delta(records: list, table_name: str):
    df = spark.createDataFrame(pd.DataFrame([dataclasses.asdict(r) for r in records]))
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{table_name}")
    print(f"  {len(records):>4} records  ->  {catalog}.{schema}.{table_name}")
    return df

bridges_df = _save_to_delta(bridges, "model_bridges")
drift_df   = _save_to_delta(drift,   "model_drift")

# COMMAND ----------
# MAGIC %md
# MAGIC ## KPI Summary

# COMMAND ----------

total_holdings = len(holdings)
total_fv       = drift[-1].total_portfolio_nav_mm
flagged_count  = sum(1 for a in assumptions if a.approval_status == "flagged")
changed_count  = sum(1 for a in assumptions if a.assumption_changed)
cum_op         = sum(d.operational_change_mm for d in drift[1:])
cum_assump     = sum(d.assumption_change_mm  for d in drift[1:])
cum_total      = cum_op + cum_assump
assump_pct     = abs(cum_assump) / abs(cum_total) * 100 if cum_total else 0

kpi_warn = WARN if assump_pct > 30 or flagged_count > 0 else ACCENT

# Assumption source coverage (latest quarter)
latest = drift[-1]
src_total = latest.nav_disclosed_mm + latest.nav_back_calculated_mm + latest.nav_extracted_mm
if src_total > 0:
    pct_disclosed = latest.nav_disclosed_mm / src_total * 100
    pct_back_calc = latest.nav_back_calculated_mm / src_total * 100
    pct_extracted = latest.nav_extracted_mm / src_total * 100
else:
    pct_disclosed = pct_back_calc = pct_extracted = 0.0

displayHTML(f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            display:grid;grid-template-columns:repeat(6,1fr);gap:12px;padding:20px 0;">
  <div style="background:white;border-left:4px solid {ACCENT};padding:14px 18px;border-radius:0 6px 6px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);">
    <div style="font-size:1.6em;font-weight:700;color:{ACCENT}">{total_holdings}</div>
    <div style="font-size:0.9em;color:{NAVY};font-weight:600;margin-top:2px">Holdings</div>
    <div style="font-size:0.78em;color:{GREY};margin-top:4px">Buyout · Growth Equity · Infra · RE</div>
  </div>
  <div style="background:white;border-left:4px solid {ACCENT};padding:14px 18px;border-radius:0 6px 6px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);">
    <div style="font-size:1.6em;font-weight:700;color:{ACCENT}">${total_fv:,.0f}M</div>
    <div style="font-size:0.9em;color:{NAVY};font-weight:600;margin-top:2px">Portfolio NAV (Q3 2025)</div>
    <div style="font-size:0.78em;color:{GREY};margin-top:4px">Mark-to-model fair values</div>
  </div>
  <div style="background:white;border-left:4px solid {kpi_warn};padding:14px 18px;border-radius:0 6px 6px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);">
    <div style="font-size:1.6em;font-weight:700;color:{kpi_warn}">{assump_pct:.0f}%</div>
    <div style="font-size:0.9em;color:{NAVY};font-weight:600;margin-top:2px">Assumption-driven NAV change</div>
    <div style="font-size:0.78em;color:{GREY};margin-top:4px">of cumulative change over 7 quarters</div>
  </div>
  <div style="background:white;border-left:4px solid {ACCENT};padding:14px 18px;border-radius:0 6px 6px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);">
    <div style="font-size:1.6em;font-weight:700;color:{ACCENT}">{changed_count}</div>
    <div style="font-size:0.9em;color:{NAVY};font-weight:600;margin-top:2px">Assumption changes logged</div>
    <div style="font-size:0.78em;color:{GREY};margin-top:4px">Tracked per holding per quarter</div>
  </div>
  <div style="background:white;border-left:4px solid {kpi_warn};padding:14px 18px;border-radius:0 6px 6px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);">
    <div style="font-size:1.6em;font-weight:700;color:{kpi_warn}">{flagged_count}</div>
    <div style="font-size:0.9em;color:{NAVY};font-weight:600;margin-top:2px">Flagged for review</div>
    <div style="font-size:0.78em;color:{GREY};margin-top:4px">Pending board approval</div>
  </div>
  <div style="background:white;border-left:4px solid {ACCENT};padding:14px 18px;border-radius:0 6px 6px 0;box-shadow:0 1px 4px rgba(0,0,0,.08);">
    <div style="font-size:1.4em;font-weight:700;color:{ACCENT}">{pct_disclosed:.0f}% / {pct_back_calc:.0f}% / {pct_extracted:.0f}%</div>
    <div style="font-size:0.9em;color:{NAVY};font-weight:600;margin-top:2px">Assumption Data Sources</div>
    <div style="font-size:0.78em;color:{GREY};margin-top:4px">Disclosed / Back-calc / Extracted (by NAV)</div>
  </div>
</div>
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Scenario 1: Methodology-Driven Swing — Waterfall Bridge
# MAGIC
# MAGIC BYT-2020-I adjusted EBITDA multiples sharply upward in Q3 2024 following ECB rate cuts.
# MAGIC The bridge shows that virtually all of the NAV movement came from assumption change,
# MAGIC not from anything the underlying companies did. **Flagged for board review.**

# COMMAND ----------

def chart_waterfall(bridges):
    fund_id = SCENARIO_SWING["fund_id"]
    quarter = SCENARIO_SWING["trigger_quarter"]

    fund_bridges = [b for b in bridges if b.fund_id == fund_id and b.quarter == quarter]
    if not fund_bridges:
        print("No bridge data for this scenario.")
        return

    fund_obj = next((f for f in FUND_DEFINITIONS if f.fund_id == fund_id), None)
    gp_name  = fund_obj.gp_name if fund_obj else fund_id

    opening     = sum(b.opening_fair_value_mm     for b in fund_bridges)
    operational = sum(b.operational_component_mm  for b in fund_bridges)
    assumption  = sum(b.assumption_component_mm   for b in fund_bridges)
    methodology = sum(b.methodology_component_mm  for b in fund_bridges)
    closing     = sum(b.closing_fair_value_mm      for b in fund_bridges)

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
    ax.text(0.98, 0.97, "FLAGGED - board review required",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=WARN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9E6", edgecolor=WARN, lw=1))
    fig.tight_layout()
    return fig

fig = chart_waterfall(bridges)
if fig:
    display(fig)
    plt.show()
    plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Scenario 3: Systematic Assumption Drift
# MAGIC
# MAGIC Across all buyout funds, the average EBITDA multiple drifted upward quarter by quarter
# MAGIC tracking public-market PE repricing during the rate-cut cycle.
# MAGIC No single quarter looks alarming. The cumulative drift is significant.

# COMMAND ----------

def chart_drift(drift):
    quarters = [d.quarter for d in drift]
    avg_mult = [d.avg_buyout_ebitda_multiple for d in drift]
    med_mult = [d.median_buyout_ebitda_multiple for d in drift]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#ffffff")

    x = range(len(quarters))
    ax.plot(x, avg_mult, color=ACCENT,  lw=2,   marker="o", ms=6, label="Average multiple",  zorder=3)
    ax.plot(x, med_mult, color=ORANGE,  lw=1.5, marker="s", ms=5, ls="--", label="Median multiple", zorder=3)

    baseline = avg_mult[0]
    ax.fill_between(x, baseline, avg_mult,
                    where=[m >= baseline for m in avg_mult],
                    alpha=0.12, color=ORANGE, label="Assumption expansion vs baseline")
    ax.axhline(baseline, color=GREY, lw=0.8, ls=":", zorder=2)
    ax.text(len(quarters) - 0.6, baseline + 0.05, f"Q1 2024 baseline: {baseline:.2f}x",
            fontsize=7.5, color=GREY, va="bottom")

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
        f"Buyout Portfolio - EBITDA Multiple Drift: {avg_mult[0]:.2f}x -> {avg_mult[-1]:.2f}x "
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
    return fig

fig = chart_drift(drift)
display(fig)
plt.show()
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Scenario 4: Cross-GP Divergence
# MAGIC
# MAGIC Two GPs hold comparable Media & Entertainment assets. Same sector, similar businesses.
# MAGIC One GP selected conservative peers; the other selected digital-premium peers.
# MAGIC The persistent multiple gap is only visible when comparing across GPs.

# COMMAND ----------

def chart_cross_gp(assumptions, holdings):
    sector   = SCENARIO_DIVERGENCE["sector"]
    fund_a   = SCENARIO_DIVERGENCE["conservative_fund"]
    fund_b   = SCENARIO_DIVERGENCE["optimistic_fund"]

    target_holdings = {h.holding_id: h for h in holdings
                       if h.sector == sector and h.fund_id in (fund_a, fund_b)}
    if not target_holdings:
        print(f"No {sector} holdings found in {fund_a} or {fund_b}.")
        return

    fund_a_series: dict = defaultdict(list)
    fund_b_series: dict = defaultdict(list)

    for q in QUARTERS:
        for hid, h in target_holdings.items():
            rec = next((a for a in assumptions if a.holding_id == hid and a.quarter == q), None)
            if rec and rec.ebitda_multiple is not None:
                if h.fund_id == fund_a:
                    fund_a_series[h.company_name].append(rec.ebitda_multiple)
                else:
                    fund_b_series[h.company_name].append(rec.ebitda_multiple)

    if not fund_a_series and not fund_b_series:
        print("No EBITDA multiples found for this sector.")
        return

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

    # Avg lines — only full-length series (matches what gets plotted)
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
        f"Cross-GP Divergence - {sector}\n"
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
    return fig

fig = chart_cross_gp(assumptions, holdings)
if fig:
    display(fig)
    plt.show()
    plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Portfolio NAV Decomposition
# MAGIC
# MAGIC Each quarter's NAV change decomposed into what the businesses earned (operational)
# MAGIC and what the valuation assumptions contributed (assumption).
# MAGIC A well-governed portfolio should show assumption changes that are explainable and approved.

# COMMAND ----------

def chart_decomposition(drift):
    bridge_quarters = QUARTERS[1:]
    drift_map       = {d.quarter: d for d in drift}

    op_vals   = [drift_map[q].operational_change_mm  for q in bridge_quarters]
    asmp_vals = [drift_map[q].assumption_change_mm   for q in bridge_quarters]
    meth_vals = [drift_map[q].methodology_change_mm  for q in bridge_quarters]

    x     = np.arange(len(bridge_quarters))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#ffffff")

    def _pos(v): return max(v, 0)
    def _neg(v): return min(v, 0)

    ax.bar(x, [_pos(v) for v in op_vals],   width, color=GREEN,  label="Operational",  zorder=3)
    ax.bar(x, [_pos(v) for v in asmp_vals],  width,
           bottom=[_pos(o) for o in op_vals],
           color=ORANGE, label="Assumption", zorder=3)
    ax.bar(x, [_pos(v) for v in meth_vals],  width,
           bottom=[_pos(o) + _pos(a) for o, a in zip(op_vals, asmp_vals)],
           color=RED, label="Methodology", zorder=3)

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
        "Portfolio NAV Change by Quarter - Operational vs Assumption vs Methodology\n"
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
    return fig

fig = chart_decomposition(drift)
display(fig)
plt.show()
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Governance Log - Flagged Assumptions
# MAGIC
# MAGIC Every assumption that triggered a review flag. In a real system this would include
# MAGIC the approver, approval date, and supporting memo reference.

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT
            a.fund_id,
            a.quarter,
            h.company_name,
            a.method,
            COALESCE(a.ebitda_multiple, a.revenue_multiple) AS multiple,
            a.assumption_source,
            a.approval_status,
            SUBSTRING(a.notes, 1, 120)                      AS notes
        FROM {catalog}.{schema}.model_assumptions a
        JOIN {catalog}.{schema}.model_holdings h
          ON a.holding_id = h.holding_id
        WHERE a.approval_status = 'flagged'
        ORDER BY a.fund_id, a.quarter
    """)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC All six Pattern 2 tables are now populated:
# MAGIC `model_holdings`, `model_assumptions`, `model_valuations`, `model_bridges`, `model_drift`, `model_data_access`
# MAGIC
# MAGIC The local HTML equivalent is at `local/output/model_governance_dashboard.html`.
