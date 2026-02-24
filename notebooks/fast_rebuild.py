# Databricks notebook source
# MAGIC %md
# MAGIC # Fast Rebuild — Tables Only (No PDFs, No LLM)
# MAGIC
# MAGIC Drops and rebuilds all Delta tables from simulation without regenerating PDFs
# MAGIC or re-running LLM extraction. Use this when iterating on:
# MAGIC - Fund definitions, simulation parameters, failure modes
# MAGIC - Ground truth schema or dashboard queries
# MAGIC - Anything that doesn't require new PDF content or extraction results
# MAGIC
# MAGIC **Skips:** PDF generation (notebook 01 Step 3), LLM extraction (notebook 02)
# MAGIC
# MAGIC **Rebuilds:** ground_truth, ground_truth_portfolio_companies, ground_truth_capital_calls,
# MAGIC ground_truth_distributions, report_manifest, data_quality_flags,
# MAGIC cost_model_results, cost_model_summary, and (if extractions exist) extraction_results,
# MAGIC extraction_summary, routing_thresholds, calibration_results, calibration_results_json
# MAGIC
# MAGIC **Preserves:** PDFs in Volume, extractions table (if it exists)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Cleanup — Drop Tables

# COMMAND ----------

# Create widgets with defaults (idempotent — preserves user's current value)
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schema", "extraction_demo", "Schema Name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

tables_to_drop = [
    # Generation tables (rebuilt by this notebook)
    "ground_truth",
    "ground_truth_portfolio_companies",
    "ground_truth_capital_calls",
    "ground_truth_distributions",
    "report_manifest",
    "data_quality_flags",
    # Downstream tables (stale once ground truth changes)
    "extraction_results",
    "extraction_summary",
    "routing_thresholds",
    "calibration_results",
    "calibration_results_json",
    "cost_model_results",
    "cost_model_summary",
]

for t in tables_to_drop:
    spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{t}")
    print(f"  Dropped {catalog}.{schema}.{t}")

print(f"\n{len(tables_to_drop)} tables dropped. PDFs and raw extractions preserved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Setup & Imports

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

volume_path = f"/Volumes/{catalog}/{schema}/reports"

# COMMAND ----------

import os
import sys

notebook_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
project_root = os.path.join(notebook_dir, "..")
sys.path.insert(0, os.path.abspath(project_root))

from shared.fund_definitions import FUND_DEFINITIONS, QUARTERS, FX_RATES, FX_RATE_DATE
from shared.simulation import simulate_fund_quarters, generate_capital_call_events, generate_distribution_events
from shared.failure_modes import (
    apply_timing_variation,
    apply_silent_restatements,
    apply_transcription_errors,
    apply_irr_ambiguity,
)

print(f"Funds: {len(FUND_DEFINITIONS)}")
print(f"Quarters: {len(QUARTERS)} ({QUARTERS[0]} to {QUARTERS[-1]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Simulate & Apply Failure Modes

# COMMAND ----------

all_snapshots = {}
all_capital_calls = {}
all_distributions = {}

for fund_def in FUND_DEFINITIONS:
    all_snapshots[fund_def.fund_id] = simulate_fund_quarters(fund_def, QUARTERS)
    all_capital_calls[fund_def.fund_id] = generate_capital_call_events(fund_def, QUARTERS)
    all_distributions[fund_def.fund_id] = generate_distribution_events(fund_def, QUARTERS)

total_snaps = sum(len(s) for s in all_snapshots.values())
total_calls = sum(len(c) for c in all_capital_calls.values())
total_dists = sum(len(d) for d in all_distributions.values())
print(f"{total_snaps} quarter-snapshots generated")
print(f"{total_calls} capital call events generated")
print(f"{total_dists} distribution events generated")

# COMMAND ----------

all_snapshots = apply_timing_variation(all_snapshots)
all_snapshots, restated_snapshots = apply_silent_restatements(all_snapshots)
transcription_errors = apply_transcription_errors(all_snapshots)
irr_ambiguous_docs = apply_irr_ambiguity(all_snapshots)

carry_forward = sum(
    1 for snaps in all_snapshots.values()
    for s in snaps if s.actual_or_estimated == "carry_forward"
)
actual_reports = sum(
    1 for snaps in all_snapshots.values()
    for s in snaps if s.actual_or_estimated == "actual"
)

print(f"Actual reports: {actual_reports}")
print(f"Carry-forward (missing): {carry_forward}")
print(f"Restated rows: {len(restated_snapshots)}")
print(f"Transcription errors: {len(transcription_errors)} reports")
print(f"IRR ambiguous: {len(irr_ambiguous_docs)} reports")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Write Delta Tables — All Document Types

# COMMAND ----------

fund_def_lookup = {fd.fund_id: fd for fd in FUND_DEFINITIONS}

# Build ground truth rows for ALL document types
ground_truth_rows = []
gt_pc_rows = []
gt_cc_rows = []
gt_dn_rows = []
manifest_rows = []

for fund_id, snapshots in all_snapshots.items():
    fund_def = fund_def_lookup[fund_id]

    for snap in snapshots:
        if snap.actual_or_estimated == "not_yet_active":
            continue

        ground_truth_rows.append({
            "fund_id": snap.fund_id,
            "fund_name": snap.fund_name,
            "gp_name": snap.gp_name,
            "reporting_period": snap.reporting_period,
            "quarter_end_date": snap.quarter_end_date,
            "report_date": snap.report_date,
            "source_document_id": snap.source_document_id,
            "actual_or_estimated": snap.actual_or_estimated,
            "document_type": "quarterly_report",
            "vintage_year": snap.vintage_year,
            "strategy": snap.strategy,
            "currency": snap.currency,
            "fx_rate_to_usd": float(snap.fx_rate_to_usd),
            "fx_rate_date": snap.fx_rate_date,
            "committed_capital_mm": float(snap.committed_capital_mm),
            "called_capital_mm": float(snap.called_capital_mm),
            "distributed_capital_mm": float(snap.distributed_capital_mm),
            "nav_mm": float(snap.nav_mm),
            "net_irr_pct": float(snap.net_irr_pct),
            "gross_irr_pct": float(snap.gross_irr_pct),
            "tvpi": float(snap.tvpi),
            "dpi": float(snap.dpi),
            "rvpi": float(snap.rvpi),
            "management_fee_mm": float(snap.management_fee_mm),
            "carried_interest_mm": float(snap.carried_interest_mm),
            "other_expenses_mm": float(snap.other_expenses_mm),
            "report_quality_tier": snap.report_quality_tier,
            "num_portfolio_companies": len(snap.portfolio_companies),
        })

        for pc in snap.portfolio_companies:
            gt_pc_rows.append({
                "fund_id": snap.fund_id,
                "reporting_period": snap.reporting_period,
                "actual_or_estimated": snap.actual_or_estimated,
                "company_name": pc.name,
                "sector": pc.sector,
                "investment_date": pc.investment_date,
                "initial_cost_mm": float(pc.initial_cost_mm),
                "fair_value_mm": float(pc.fair_value_mm),
            })

        if snap.actual_or_estimated == "actual" and snap.source_document_id:
            manifest_rows.append({
                "fund_id": snap.fund_id,
                "fund_name": snap.fund_name,
                "reporting_period": snap.reporting_period,
                "report_quality_tier": snap.report_quality_tier,
                "source_document_id": snap.source_document_id,
                "document_type": "quarterly_report",
                "path": os.path.join(volume_path, snap.source_document_id),
            })

# Capital call ground truth — matches notebook 01 structure exactly
for fund_id, calls in all_capital_calls.items():
    for call in calls:
        gt_row = {
            "document_type": "capital_call",
            "fund_id": call.fund_id,
            "fund_name": call.fund_name,
            "gp_name": call.gp_name,
            "vintage_year": call.vintage_year,
            "strategy": call.strategy,
            "currency": call.currency,
            "committed_capital_mm": float(call.committed_capital_mm),
            "lp_commitment_mm": float(call.lp_commitment_mm),
            "call_date": call.call_date,
            "due_date": call.due_date,
            "call_amount_mm": float(call.call_amount_mm),
            "call_amount_pct": float(call.call_amount_pct),
            "cumulative_called_mm": float(call.cumulative_called_mm),
            "unfunded_commitment_mm": float(call.unfunded_commitment_mm),
            "bank_name": call.bank_name,
            "account_name": call.account_name,
            "account_number": call.account_number,
            "routing_number": call.routing_number,
            "swift_code": call.swift_code,
            "iban": call.iban,
            "lp_commitment_reference": call.lp_commitment_reference,
            "report_quality_tier": call.report_quality_tier,
            "source_document_id": call.source_document_id,
            "terminology": call.terminology,
        }
        gt_cc_rows.append(gt_row)
        ground_truth_rows.append(gt_row)
        manifest_rows.append({
            "document_type": "capital_call",
            "fund_id": call.fund_id,
            "fund_name": call.fund_name,
            "reporting_period": call.call_date,
            "report_quality_tier": call.report_quality_tier,
            "source_document_id": call.source_document_id,
            "path": os.path.join(volume_path, call.source_document_id),
        })

# Distribution ground truth — matches notebook 01 structure exactly
for fund_id, dists in all_distributions.items():
    for dist in dists:
        gt_row = {
            "document_type": "distribution",
            "fund_id": dist.fund_id,
            "fund_name": dist.fund_name,
            "gp_name": dist.gp_name,
            "vintage_year": dist.vintage_year,
            "strategy": dist.strategy,
            "currency": dist.currency,
            "committed_capital_mm": float(dist.committed_capital_mm),
            "lp_commitment_mm": float(dist.lp_commitment_mm),
            "distribution_date": dist.distribution_date,
            "distribution_amount_mm": float(dist.distribution_amount_mm),
            "distribution_type": dist.distribution_type,
            "cumulative_distributed_mm": float(dist.cumulative_distributed_mm),
            "realization_source": dist.realization_source,
            "lp_commitment_reference": dist.lp_commitment_reference,
            "report_quality_tier": dist.report_quality_tier,
            "source_document_id": dist.source_document_id,
            "terminology": dist.terminology,
        }
        gt_dn_rows.append(gt_row)
        ground_truth_rows.append(gt_row)
        manifest_rows.append({
            "document_type": "distribution",
            "fund_id": dist.fund_id,
            "fund_name": dist.fund_name,
            "reporting_period": dist.distribution_date,
            "report_quality_tier": dist.report_quality_tier,
            "source_document_id": dist.source_document_id,
            "path": os.path.join(volume_path, dist.source_document_id),
        })

# Add restated rows
for snap in restated_snapshots:
    ground_truth_rows.append({
        "fund_id": snap.fund_id,
        "fund_name": snap.fund_name,
        "gp_name": snap.gp_name,
        "reporting_period": snap.reporting_period,
        "quarter_end_date": snap.quarter_end_date,
        "report_date": snap.report_date,
        "source_document_id": snap.source_document_id,
        "actual_or_estimated": snap.actual_or_estimated,
        "document_type": "quarterly_report",
        "vintage_year": snap.vintage_year,
        "strategy": snap.strategy,
        "currency": snap.currency,
        "fx_rate_to_usd": float(snap.fx_rate_to_usd),
        "fx_rate_date": snap.fx_rate_date,
        "committed_capital_mm": float(snap.committed_capital_mm),
        "called_capital_mm": float(snap.called_capital_mm),
        "distributed_capital_mm": float(snap.distributed_capital_mm),
        "nav_mm": float(snap.nav_mm),
        "net_irr_pct": float(snap.net_irr_pct),
        "gross_irr_pct": float(snap.gross_irr_pct),
        "tvpi": float(snap.tvpi),
        "dpi": float(snap.dpi),
        "rvpi": float(snap.rvpi),
        "management_fee_mm": float(snap.management_fee_mm),
        "carried_interest_mm": float(snap.carried_interest_mm),
        "other_expenses_mm": float(snap.other_expenses_mm),
        "report_quality_tier": snap.report_quality_tier,
        "num_portfolio_companies": len(snap.portfolio_companies),
    })

print(f"Ground truth rows: {len(ground_truth_rows)} (all document types)")
print(f"Portfolio company rows: {len(gt_pc_rows)}")
print(f"Capital call rows: {len(gt_cc_rows)}")
print(f"Distribution rows: {len(gt_dn_rows)}")
print(f"Manifest entries: {len(manifest_rows)}")

# COMMAND ----------

# Write ground truth (schema inferred — avoids field-name mismatch bugs)
gt_df = spark.createDataFrame(ground_truth_rows)
gt_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth")
print(f"Wrote {gt_df.count()} rows to {catalog}.{schema}.ground_truth")

# COMMAND ----------

pc_df = spark.createDataFrame(gt_pc_rows)
pc_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth_portfolio_companies")
print(f"Wrote {pc_df.count()} rows to {catalog}.{schema}.ground_truth_portfolio_companies")

# COMMAND ----------

if gt_cc_rows:
    cc_df = spark.createDataFrame(gt_cc_rows)
    cc_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth_capital_calls")
    print(f"Wrote {cc_df.count()} rows to {catalog}.{schema}.ground_truth_capital_calls")
else:
    print("No capital call events generated")

if gt_dn_rows:
    dn_df = spark.createDataFrame(gt_dn_rows)
    dn_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth_distributions")
    print(f"Wrote {dn_df.count()} rows to {catalog}.{schema}.ground_truth_distributions")
else:
    print("No distribution events generated")

# COMMAND ----------

manifest_df = spark.createDataFrame(manifest_rows)
manifest_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.report_manifest")
print(f"Wrote {manifest_df.count()} rows to {catalog}.{schema}.report_manifest")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Flags

# COMMAND ----------

flag_rows = []
for doc_id, errs in transcription_errors.items():
    err_parts = []
    if "tvpi" in errs:
        err_parts.append(f"TVPI shown as {errs['tvpi']:.2f}x")
    pc_errs = [k for k in errs if k.startswith("pc_fair_value_")]
    if pc_errs:
        err_parts.append("Portfolio company sum mismatch")
    flag_rows.append({
        "source_document_id": doc_id,
        "flag_type": "Transcription Error",
        "details": "; ".join(err_parts),
    })

for doc_id in sorted(irr_ambiguous_docs):
    flag_rows.append({
        "source_document_id": doc_id,
        "flag_type": "IRR Ambiguity",
        "details": "Gross/Net IRR not clearly labelled",
    })

flag_df = spark.createDataFrame(flag_rows)
flag_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.data_quality_flags")
print(f"Wrote {flag_df.count()} rows to {catalog}.{schema}.data_quality_flags")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Cost Model — Multi-Document Type Support (always runs — pure math)

# COMMAND ----------

import numpy as np

# Cost parameters matching notebook 04 — multi-document type
RULES_FIXED_COST = 80_000
RULES_PER_LAYOUT_COST = 3_000
RULES_ANNUAL_MAINT_PER_LAYOUT = 1_200

LLM_FIXED_COST = 150_000
LLM_PER_COUNTERPARTY_COST = 500
LLM_ANNUAL_MAINTENANCE = 40_000
LLM_API_COST_PER_DOC = 0.50

# Document volume per counterparty per year (same as notebook 04)
QUARTERLY_REPORTS_PER_YEAR = 4
CAPITAL_CALLS_PER_YEAR = 6
DISTRIBUTIONS_PER_YEAR = 3
TOTAL_DOCS_PER_COUNTERPARTY_YEAR = QUARTERLY_REPORTS_PER_YEAR + CAPITAL_CALLS_PER_YEAR + DISTRIBUTIONS_PER_YEAR

# Document type complexity
LAYOUTS_PER_COUNTERPARTY = {"quarterly_reports": 1, "capital_calls": 1.5, "distributions": 1.2}
TOTAL_LAYOUTS_PER_COUNTERPARTY = sum(LAYOUTS_PER_COUNTERPARTY.values())

max_cp = 500
counterparties = np.arange(1, max_cp + 1)

# Year 1 — multi-document
total_layouts = TOTAL_LAYOUTS_PER_COUNTERPARTY * counterparties
total_docs = TOTAL_DOCS_PER_COUNTERPARTY_YEAR * counterparties
rules_year1 = RULES_FIXED_COST + (RULES_PER_LAYOUT_COST * total_layouts) + (RULES_ANNUAL_MAINT_PER_LAYOUT * total_layouts)
llm_year1 = LLM_FIXED_COST + (LLM_PER_COUNTERPARTY_COST * counterparties) + LLM_ANNUAL_MAINTENANCE + (LLM_API_COST_PER_DOC * total_docs)

crossover_idx = np.argmax(rules_year1 > llm_year1)
crossover_n = counterparties[crossover_idx] if rules_year1[crossover_idx] > llm_year1[crossover_idx] else None

# 3-Year TCO with 10% growth
years = 3
growth_rate = 0.10
rules_3yr = np.zeros_like(counterparties, dtype=float)
llm_3yr = np.zeros_like(counterparties, dtype=float)

for year in range(years):
    effective_cp = counterparties * (1 + growth_rate) ** year
    effective_layouts = TOTAL_LAYOUTS_PER_COUNTERPARTY * effective_cp
    effective_docs = TOTAL_DOCS_PER_COUNTERPARTY_YEAR * effective_cp
    if year == 0:
        rules_3yr += RULES_FIXED_COST + (RULES_PER_LAYOUT_COST * total_layouts) + (RULES_ANNUAL_MAINT_PER_LAYOUT * total_layouts)
        llm_3yr += LLM_FIXED_COST + (LLM_PER_COUNTERPARTY_COST * counterparties) + LLM_ANNUAL_MAINTENANCE + (LLM_API_COST_PER_DOC * total_docs)
    else:
        new_cp = counterparties * growth_rate
        new_layouts = TOTAL_LAYOUTS_PER_COUNTERPARTY * new_cp
        rules_3yr += (RULES_PER_LAYOUT_COST * new_layouts) + (RULES_ANNUAL_MAINT_PER_LAYOUT * effective_layouts)
        llm_3yr += (LLM_PER_COUNTERPARTY_COST * new_cp) + LLM_ANNUAL_MAINTENANCE + (LLM_API_COST_PER_DOC * effective_docs)

crossover_3yr_idx = np.argmax(rules_3yr > llm_3yr)
crossover_3yr_n = counterparties[crossover_3yr_idx] if rules_3yr[crossover_3yr_idx] > llm_3yr[crossover_3yr_idx] else None

# Write cost_model_results (sampled every 10th)
sample_step = 10
sample_indices = list(range(0, max_cp, sample_step))
cost_rows = [
    {
        "counterparties": int(counterparties[i]),
        "rules_year1": float(rules_year1[i]),
        "llm_year1": float(llm_year1[i]),
        "rules_3yr": float(rules_3yr[i]),
        "llm_3yr": float(llm_3yr[i]),
    }
    for i in sample_indices
]

cost_sdf = spark.createDataFrame(cost_rows)
cost_sdf.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.cost_model_results")
print(f"Wrote {cost_sdf.count()} rows to {catalog}.{schema}.cost_model_results")

# Crossover summary (schema matches notebook 04: period + crossover_counterparties only)
summary_rows = []
if crossover_n:
    summary_rows.append({"period": "Year 1", "crossover_counterparties": int(crossover_n)})
if crossover_3yr_n:
    summary_rows.append({"period": "3-Year TCO", "crossover_counterparties": int(crossover_3yr_n)})

if summary_rows:
    cs_sdf = spark.createDataFrame(summary_rows)
    cs_sdf.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.cost_model_summary")
    print(f"Wrote crossover summary to {catalog}.{schema}.cost_model_summary")

if crossover_n:
    print(f"Year 1 crossover: {crossover_n} counterparties")
if crossover_3yr_n:
    print(f"3-year crossover: {crossover_3yr_n} counterparties")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Calibration & Routing (if extractions exist)
# MAGIC
# MAGIC Runs the notebook 03 logic: accuracy measurement per document type, isotonic
# MAGIC calibration, routing threshold sweep with document-type-specific thresholds.
# MAGIC Only runs if the `extractions` table exists from a prior notebook 02 run.

# COMMAND ----------

# Check if extractions table exists
extractions_exist = False
try:
    ext_count = spark.table(f"{catalog}.{schema}.extractions").count()
    extractions_exist = ext_count > 0
    print(f"Found {ext_count} rows in extractions table — running calibration & routing")
except Exception:
    print("No extractions table found — skipping calibration & routing (run notebook 02 first)")

# COMMAND ----------

if extractions_exist:
    # Install scikit-learn if not available (pre-installed on ML Runtime)
    import importlib.util
    if importlib.util.find_spec("sklearn") is None:
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-learn"])

    import json as _json
    from collections import defaultdict
    from sklearn.isotonic import IsotonicRegression
    from shared.extraction_utils import reconstruct_extractions

    # Load data
    extractions_flat = spark.table(f"{catalog}.{schema}.extractions").toPandas().to_dict("records")
    ground_truth_all = spark.table(f"{catalog}.{schema}.ground_truth").toPandas().to_dict("records")

    # Reconstruct nested extraction dicts (shared utility — SSOT)
    extractions = reconstruct_extractions(extractions_flat)
    print(f"Reconstructed {len(extractions)} document extractions from {len(extractions_flat)} field rows")

    # Load CC/DN ground truth if available
    gt_cc_loaded = []
    gt_dn_loaded = []
    try:
        gt_cc_loaded = spark.table(f"{catalog}.{schema}.ground_truth_capital_calls").toPandas().to_dict("records")
    except Exception:
        pass
    try:
        gt_dn_loaded = spark.table(f"{catalog}.{schema}.ground_truth_distributions").toPandas().to_dict("records")
    except Exception:
        pass

    # Index ground truth by document type and key
    gt_index_qr = {}
    gt_index_cc = {}
    gt_index_dn = {}

    for row in ground_truth_all:
        doc_type = row.get("document_type", "quarterly_report")
        if doc_type == "quarterly_report" and row.get("actual_or_estimated") == "actual":
            gt_index_qr[(row["fund_id"], row["reporting_period"])] = row

    for row in gt_cc_loaded:
        gt_index_cc[(row["fund_id"], row["call_date"])] = row

    for row in gt_dn_loaded:
        gt_index_dn[(row["fund_id"], row["distribution_date"])] = row

    gt_indices = {
        "quarterly_report": gt_index_qr,
        "capital_call": gt_index_cc,
        "distribution": gt_index_dn,
    }

    # Import extraction schemas and routing thresholds from shared SSOT
    from shared.schemas import SCHEMAS, ROUTING_THRESHOLDS

    def check_accuracy(extracted_value, true_value, field_type):
        NUMERIC_TOLERANCE = 0.01
        if extracted_value is None and true_value is None:
            return True
        if extracted_value is None or true_value is None:
            return False
        if field_type in ("number", "integer"):
            try:
                extracted_num = float(extracted_value)
                true_num = float(true_value)
                if true_num == 0:
                    return abs(extracted_num) < 0.01
                return abs(extracted_num - true_num) / abs(true_num) <= NUMERIC_TOLERANCE
            except (ValueError, TypeError):
                return False
        else:
            return str(extracted_value).strip().lower() == str(true_value).strip().lower()

    def get_routing_decision(calibrated_confidence, doc_type):
        thresholds = ROUTING_THRESHOLDS[doc_type]
        if calibrated_confidence >= thresholds["high_confidence"]:
            return "auto_accept"
        elif calibrated_confidence >= thresholds["medium_confidence"]:
            return "senior_review"
        elif calibrated_confidence >= thresholds["low_confidence"]:
            return "expert_review"
        else:
            return "reject"

    # Accuracy measurement per document type
    accuracy_records = []
    for ext in extractions:
        fund_id = ext["fund_id"]
        doc_type = ext.get("document_type", "quarterly_report")
        tier = ext["report_quality_tier"]

        if doc_type == "quarterly_report":
            key = (fund_id, ext.get("reporting_period"))
        elif doc_type == "capital_call":
            key = (fund_id, ext.get("call_date", ""))
        elif doc_type == "distribution":
            key = (fund_id, ext.get("distribution_date", ""))
        else:
            continue

        gt_index = gt_indices.get(doc_type, {})
        gt_row = gt_index.get(key)
        if gt_row is None:
            continue

        field_schema = SCHEMAS.get(doc_type, SCHEMAS["quarterly_report"])
        for field_name, result in ext["extraction"].items():
            if field_name not in field_schema:
                continue
            field_type = field_schema[field_name]
            extracted_value = result.get("value")
            raw_confidence = result.get("confidence", 0.0)
            true_value = gt_row.get(field_name)
            is_correct = check_accuracy(extracted_value, true_value, field_type)

            accuracy_records.append({
                "fund_id": fund_id,
                "document_type": doc_type,
                "document_key": str(key),
                "report_quality_tier": tier,
                "field_name": field_name,
                "field_type": field_type,
                "extracted_value": str(extracted_value) if extracted_value is not None else None,
                "true_value": str(true_value) if true_value is not None else None,
                "raw_confidence": float(raw_confidence),
                "is_correct": is_correct,
                "has_source_attribution": bool(result.get("source_text")),
            })

    print(f"Accuracy records: {len(accuracy_records)}")
    if accuracy_records:
        correct = sum(1 for r in accuracy_records if r["is_correct"])
        print(f"Overall accuracy: {correct/len(accuracy_records):.1%}")

    # Isotonic calibration per document type
    calibrated_records = []
    for doc_type in ["quarterly_report", "capital_call", "distribution"]:
        doc_records = [r for r in accuracy_records if r["document_type"] == doc_type]
        if len(doc_records) < 10:
            continue

        confidences = [r["raw_confidence"] for r in doc_records]
        outcomes = [1 if r["is_correct"] else 0 for r in doc_records]

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(confidences, outcomes)
        calibrated_confidences = calibrator.predict(confidences)

        for i, record in enumerate(doc_records):
            record["calibrated_confidence"] = float(calibrated_confidences[i])
            record["routing_decision"] = get_routing_decision(calibrated_confidences[i], doc_type)
            calibrated_records.append(record)

    # Write extraction_results + calibration_results
    if calibrated_records:
        calibrated_df = spark.createDataFrame(calibrated_records)
        calibrated_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.calibration_results")
        calibrated_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.extraction_results")
        print(f"Wrote {calibrated_df.count()} rows to calibration_results + extraction_results")

    # Write extraction_summary (per-tier stats for Lakeview dashboard)
    summary_rows_cal = []
    for tier in ["institutional", "narrative", "poor"]:
        tier_records = [r for r in calibrated_records if r["report_quality_tier"] == tier]
        if not tier_records:
            continue
        n_total = len(tier_records)
        n_correct = sum(1 for r in tier_records if r["is_correct"])
        n_auto = sum(1 for r in tier_records if r["routing_decision"] == "auto_accept")
        n_auto_correct = sum(1 for r in tier_records if r["routing_decision"] == "auto_accept" and r["is_correct"])
        # Weighted average threshold across document types present in this tier
        doc_type_counts = defaultdict(int)
        for r in tier_records:
            doc_type_counts[r["document_type"]] += 1
        weighted_threshold = sum(
            ROUTING_THRESHOLDS[dt]["high_confidence"] * count
            for dt, count in doc_type_counts.items()
        ) / n_total
        summary_rows_cal.append({
            "report_quality_tier": tier,
            "total_fields": n_total,
            "overall_accuracy": n_correct / n_total if n_total else 0,
            "auto_accept_pct": n_auto / n_total if n_total else 0,
            "auto_accept_accuracy": n_auto_correct / n_auto if n_auto else 0,
            "routing_threshold": weighted_threshold,
        })

    if summary_rows_cal:
        summary_sdf = spark.createDataFrame(summary_rows_cal)
        summary_sdf.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.extraction_summary")
        print(f"Wrote {len(summary_rows_cal)} rows to extraction_summary")

    # Write routing_thresholds (sweep data for Lakeview)
    threshold_results = []
    for t in np.arange(0.50, 1.00, 0.05):
        auto = [r for r in calibrated_records if r["calibrated_confidence"] >= t]
        review = [r for r in calibrated_records if r["calibrated_confidence"] < t]
        n_auto = len(auto)
        n_auto_correct = sum(1 for r in auto if r["is_correct"])
        threshold_results.append({
            "threshold": round(float(t), 2),
            "auto_accept_accuracy": n_auto_correct / n_auto if n_auto else 0,
            "human_review_pct": len(review) / len(calibrated_records) if calibrated_records else 0,
            "errors_passed": n_auto - n_auto_correct,
        })

    if threshold_results:
        threshold_sdf = spark.createDataFrame(threshold_results)
        threshold_sdf.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.routing_thresholds")
        print(f"Wrote {threshold_sdf.count()} rows to routing_thresholds")

    # Write calibration_results_json (full results for notebook 05)
    routing_analysis = {}
    for doc_type in ["quarterly_report", "capital_call", "distribution"]:
        doc_records = [r for r in calibrated_records if r["document_type"] == doc_type]
        if not doc_records:
            continue
        routing_counts = defaultdict(int)
        for r in doc_records:
            routing_counts[r["routing_decision"]] += 1
        total_d = len(doc_records)
        routing_analysis[doc_type] = {
            "total_extractions": total_d,
            "auto_accept": routing_counts["auto_accept"],
            "senior_review": routing_counts["senior_review"],
            "expert_review": routing_counts["expert_review"],
            "reject": routing_counts["reject"],
            "auto_accept_pct": routing_counts["auto_accept"] / total_d * 100,
            "senior_review_pct": routing_counts["senior_review"] / total_d * 100,
            "expert_review_pct": routing_counts["expert_review"] / total_d * 100,
            "reject_pct": routing_counts["reject"] / total_d * 100,
        }

    full_results = {
        "summary": {
            "total_extractions": len(calibrated_records),
            "document_type_breakdown": {
                dt: {
                    "extractions": len([r for r in calibrated_records if r["document_type"] == dt]),
                    "accuracy": sum(1 for r in calibrated_records if r["document_type"] == dt and r["is_correct"]) / max(1, len([r for r in calibrated_records if r["document_type"] == dt]))
                }
                for dt in ["quarterly_report", "capital_call", "distribution"]
            }
        },
        "calibration_by_document_type": {},
        "routing_thresholds": ROUTING_THRESHOLDS,
        "routing_analysis_by_document_type": routing_analysis,
        "detailed_records": calibrated_records,
    }
    results_json = _json.dumps(full_results, default=str)
    results_json_df = spark.createDataFrame([{"results_json": results_json}])
    results_json_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.calibration_results_json")
    print(f"Wrote full calibration results JSON to calibration_results_json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Preview

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ground Truth Sample

# COMMAND ----------

display(
    spark.table(f"{catalog}.{schema}.ground_truth")
    .select("fund_id", "reporting_period", "actual_or_estimated", "document_type", "nav_mm", "tvpi", "dpi", "rvpi", "currency", "report_quality_tier")
    .orderBy("fund_id", "reporting_period")
    .limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary

# COMMAND ----------

print(f"{'='*60}")
print(f"FAST REBUILD COMPLETE — MULTI-DOCUMENT TYPE SUPPORT")
print(f"{'='*60}")
print(f"  Ground truth rows:    {len(ground_truth_rows)}")
print(f"    - actual:           {actual_reports}")
print(f"    - carry_forward:    {carry_forward}")
print(f"    - restated:         {len(restated_snapshots)}")
print(f"  Portfolio companies:  {len(gt_pc_rows)}")
print(f"  Capital calls:        {len(gt_cc_rows)}")
print(f"  Distributions:        {len(gt_dn_rows)}")
print(f"  Data quality flags:   {len(flag_rows)}")
print(f"  Cost model:           REBUILT (multi-document parameters)")
if crossover_n:
    print(f"    - Year 1 crossover: {crossover_n} counterparties")
if extractions_exist:
    print(f"  Calibration/routing:  REBUILT from existing extractions")
else:
    print(f"  Calibration/routing:  SKIPPED (no extractions table)")
print(f"  PDFs:                 NOT regenerated (existing preserved)")
print(f"  LLM extraction:       NOT re-run (existing preserved)")
print(f"{'='*60}")
