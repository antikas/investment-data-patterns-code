# Databricks notebook source
"""
07 - Generate Valuation History (Mark-to-Model Governance)

Generates synthetic mark-to-model valuation data for all mark-to-model strategies
(Buyout, Growth Equity, Infrastructure, Real Estate): 19 funds, ~60 holdings,
7 quarters (Q1 2024 - Q3 2025).

Four governance scenarios embedded in the data:
  1. Methodology swing:     BYT-2020-I, Q3 2024 - assumption-driven NAV move, flagged
  2. Hidden outperformance: GEQ-2021-I/Helios, Q2 2025 - strong ops masked by tightening
  3. Systematic drift:      all buyout funds +0.10x/quarter (tracks public PE repricing)
  4. Cross-GP divergence:   BYT-2019-I vs BYT-2022-I, same sector, different comp sets

Output tables (Delta, Unity Catalog):
  model_holdings      - static holding metadata
  model_assumptions   - per-holding per-quarter assumption snapshot
  model_valuations    - per-holding per-quarter fair value
  model_data_access   - per-fund assumption source and access mechanism

model_bridges and model_drift are produced by notebook 08.

Local equivalent: local/07_generate_valuation_history.py
No LLM calls. No external dependencies beyond PySpark.
"""

# COMMAND ----------
# MAGIC %md
# MAGIC # 07 - Generate Valuation History
# MAGIC
# MAGIC Builds the synthetic mark-to-model dataset for Pattern 2: Mark-to-Model Governance.
# MAGIC
# MAGIC **Output tables:**
# MAGIC - `model_holdings` - static per-holding metadata (fund, company, sector, cost basis)
# MAGIC - `model_assumptions` - per-holding per-quarter valuation assumptions (method, multiples, rates, approval status)
# MAGIC - `model_valuations` - per-holding per-quarter fair value derived from above
# MAGIC - `model_data_access` - per-fund assumption source (disclosed/back-calculated/extracted) and access mechanism
# MAGIC
# MAGIC **Four governance scenarios** are embedded deterministically. Run notebook 08 to compute bridges and see them.

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

import pandas as pd

notebook_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
sys.path.insert(0, project_root)

from shared.fund_definitions import QUARTERS
from shared.valuation_models import (
    DATA_ACCESS_RECORDS,
    MARK_TO_MODEL_STRATEGIES,
    SCENARIO_SWING,
    SCENARIO_HIDDEN,
    SCENARIO_DIVERGENCE,
    generate_holdings,
    generate_valuation_history,
)

print(f"Strategies: {', '.join(sorted(MARK_TO_MODEL_STRATEGIES))}")
print(f"Quarters:   {len(QUARTERS)} ({QUARTERS[0]} to {QUARTERS[-1]})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Generate Holdings

# COMMAND ----------

holdings = generate_holdings()

strategy_counts = {}
for h in holdings:
    strategy_counts[h.strategy] = strategy_counts.get(h.strategy, 0) + 1

for strategy, count in sorted(strategy_counts.items()):
    print(f"  {strategy}: {count} holdings")
print(f"  Total: {len(holdings)} holdings x {len(QUARTERS)} quarters")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Generate Valuation History

# COMMAND ----------

assumptions, valuations = generate_valuation_history(holdings)

print(f"  {len(assumptions)} assumption records")
print(f"  {len(valuations)} valuation records")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write to Delta Tables

# COMMAND ----------

def _save_to_delta(records: list, table_name: str):
    """Serialise dataclass list -> pandas -> Spark DataFrame -> Delta table (overwrite)."""
    df = spark.createDataFrame(pd.DataFrame([dataclasses.asdict(r) for r in records]))
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{table_name}")
    print(f"  {len(records):>4} records  ->  {catalog}.{schema}.{table_name}")
    return df

holdings_df      = _save_to_delta(holdings,            "model_holdings")
assumptions_df   = _save_to_delta(assumptions,         "model_assumptions")
valuations_df    = _save_to_delta(valuations,          "model_valuations")
data_access_df   = _save_to_delta(DATA_ACCESS_RECORDS, "model_data_access")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Holdings

# COMMAND ----------

display(spark.table(f"{catalog}.{schema}.model_holdings"))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Assumptions (sample: flagged only)

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT * FROM {catalog}.{schema}.model_assumptions
        WHERE approval_status = 'flagged'
        ORDER BY fund_id, quarter
    """)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Scenario Verification

# COMMAND ----------

# Scenario 1: BYT-2020-I multiple swing in Q3 2024
fund_id  = SCENARIO_SWING["fund_id"]
trigger  = SCENARIO_SWING["trigger_quarter"]
prev_q   = QUARTERS[QUARTERS.index(trigger) - 1]

fund_assumps = [a for a in assumptions if a.fund_id == fund_id]
prev_mults   = [a.ebitda_multiple for a in fund_assumps if a.quarter == prev_q   and a.ebitda_multiple is not None]
curr_mults   = [a.ebitda_multiple for a in fund_assumps if a.quarter == trigger  and a.ebitda_multiple is not None]
flagged      = [a for a in fund_assumps if a.quarter == trigger and a.approval_status == "flagged"]

if prev_mults and curr_mults:
    avg_prev = sum(prev_mults) / len(prev_mults)
    avg_curr = sum(curr_mults) / len(curr_mults)
    print(f"Scenario 1: {fund_id} assumption swing ({prev_q} -> {trigger}):")
    print(f"  Avg EBITDA multiple: {avg_prev:.2f}x -> {avg_curr:.2f}x  (delta: {avg_curr - avg_prev:+.2f}x)")
    print(f"  Flagged assumptions: {len(flagged)}")

# Scenario 2: Helios Data Systems hidden outperformance
from shared.valuation_models import _holding_id

hid      = _holding_id(SCENARIO_HIDDEN["fund_id"], SCENARIO_HIDDEN["company_name"])
company  = SCENARIO_HIDDEN["company_name"]
trigger2 = SCENARIO_HIDDEN["trigger_quarter"]
prev_q2  = QUARTERS[QUARTERS.index(trigger2) - 1]

prev_v = next((v for v in valuations if v.holding_id == hid and v.quarter == prev_q2),  None)
curr_v = next((v for v in valuations if v.holding_id == hid and v.quarter == trigger2), None)
prev_a = next((a for a in assumptions if a.holding_id == hid and a.quarter == prev_q2),  None)
curr_a = next((a for a in assumptions if a.holding_id == hid and a.quarter == trigger2), None)

if prev_v and curr_v and prev_a and curr_a:
    metric_growth = (curr_v.underlying_metric_mm / prev_v.underlying_metric_mm - 1) * 100
    nav_change    = curr_v.fair_value_mm - prev_v.fair_value_mm
    mult_prev     = prev_a.revenue_multiple or 0
    mult_curr     = curr_a.revenue_multiple or 0
    print(f"\nScenario 2: {company} ({prev_q2} -> {trigger2}):")
    print(f"  Revenue:  {prev_v.underlying_metric_mm:.1f}M -> {curr_v.underlying_metric_mm:.1f}M  (+{metric_growth:.0f}%)")
    print(f"  Multiple: {mult_prev:.2f}x -> {mult_curr:.2f}x  (delta: {mult_curr - mult_prev:+.2f}x)")
    print(f"  Fair value change: {nav_change:+.1f}M  (ops masked by multiple compression)")

# Scenario 4: cross-GP divergence
sector = SCENARIO_DIVERGENCE["sector"]
fund_a = SCENARIO_DIVERGENCE["conservative_fund"]
fund_b = SCENARIO_DIVERGENCE["optimistic_fund"]
q_last = QUARTERS[-1]

hids_a = {h.holding_id for h in holdings if h.fund_id == fund_a and h.sector == sector}
hids_b = {h.holding_id for h in holdings if h.fund_id == fund_b and h.sector == sector}

mults_a = [a.ebitda_multiple for a in assumptions if a.holding_id in hids_a and a.quarter == q_last and a.ebitda_multiple is not None]
mults_b = [a.ebitda_multiple for a in assumptions if a.holding_id in hids_b and a.quarter == q_last and a.ebitda_multiple is not None]

if mults_a and mults_b:
    print(f"\nScenario 4 - Cross-GP divergence ({sector}) in {q_last}:")
    print(f"  {fund_a} (conservative peers): avg {sum(mults_a)/len(mults_a):.2f}x EBITDA")
    print(f"  {fund_b} (growth peers):        avg {sum(mults_b)/len(mults_b):.2f}x EBITDA")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC Four tables written. Run **notebook 08** to compute attribution bridges and view the governance dashboard.
