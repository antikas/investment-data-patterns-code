# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Generate Synthetic GP Documents (PDF)
# MAGIC
# MAGIC Creates synthetic **quarterly reports**, **capital call notices**, and **distribution notices** as PDFs across three quality tiers:
# MAGIC - **Institutional** — ILPA-standard formatting, structured tables, complete fields
# MAGIC - **Narrative** — Numbers embedded in prose, commentary, less structured
# MAGIC - **Poor quality** — Bare-bones, inconsistent formatting, missing fields
# MAGIC
# MAGIC **Dataset:** ~27 funds × 7 quarters (Q1 2024 – Q3 2025), with J-curve progression, multi-currency, and all 7 GP report failure modes.
# MAGIC
# MAGIC **Output:**
# MAGIC - PDFs in Unity Catalog Volume: `/Volumes/{catalog}/{schema}/reports/`
# MAGIC - Delta tables: `ground_truth`, `ground_truth_portfolio_companies`, `ground_truth_capital_calls`, `ground_truth_distributions`, `report_manifest`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Check and install dependencies not available on the base runtime
import importlib.util, subprocess

_missing = [pkg for pkg, mod in [("reportlab", "reportlab")] if importlib.util.find_spec(mod) is None]
if _missing:
    subprocess.check_call(["pip", "install"] + _missing)
    dbutils.library.restartPython()

# COMMAND ----------

# Create widgets with defaults (idempotent — preserves user's current value)
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schema", "extraction_demo", "Schema Name")
dbutils.widgets.text("volume", "reports", "Volume Name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
print(f"PDFs will be written to: {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Shared Modules

# COMMAND ----------

import os
import sys

# Add project root to path so we can import shared/
# In Databricks Repos, the notebook is at /Workspace/Repos/<user>/<repo>/notebooks/
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
from shared.report_generators import GENERATORS
from shared.notice_generators import CAPITAL_CALL_GENERATORS, DISTRIBUTION_GENERATORS

print(f"Funds: {len(FUND_DEFINITIONS)}")
print(f"Quarters: {len(QUARTERS)} ({QUARTERS[0]} to {QUARTERS[-1]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Simulate Fund Quarters

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

# MAGIC %md
# MAGIC ## Step 2: Apply Failure Modes

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

# Apply failure modes to capital calls and distribution notices
import random
rng = random.Random(42)

call_transcription_errors = {}
for fund_id, calls in all_capital_calls.items():
    for call in calls:
        if rng.random() < 0.1:
            call_transcription_errors[call.source_document_id] = {
                "call_amount_mm": call.call_amount_mm * rng.uniform(0.9, 1.1)
            }

dist_transcription_errors = {}
for fund_id, dists in all_distributions.items():
    for dist in dists:
        if rng.random() < 0.1:
            if rng.random() < 0.5:
                wrong_types = ["return_of_capital", "income", "gain"]
                wrong_types.remove(dist.distribution_type)
                dist_transcription_errors[dist.source_document_id] = {
                    "distribution_type": rng.choice(wrong_types)
                }
            else:
                dist_transcription_errors[dist.source_document_id] = {
                    "distribution_amount_mm": dist.distribution_amount_mm * rng.uniform(0.9, 1.1)
                }

print(f"Capital call transcription errors: {len(call_transcription_errors)}")
print(f"Distribution transcription errors: {len(dist_transcription_errors)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Generate PDFs

# COMMAND ----------

fund_def_lookup = {fd.fund_id: fd for fd in FUND_DEFINITIONS}
pdf_count = 0
qr_count = 0
cc_count = 0
dn_count = 0

# Quarterly report PDFs
for fund_id, snapshots in all_snapshots.items():
    fund_def = fund_def_lookup[fund_id]
    generator = GENERATORS[fund_def.report_quality_tier]

    for snap in snapshots:
        if snap.actual_or_estimated != "actual":
            continue

        filename = snap.source_document_id
        output_path = os.path.join(volume_path, filename)

        doc_errors = transcription_errors.get(filename, None)
        is_ambiguous = filename in irr_ambiguous_docs

        generator(snap, fund_def, output_path,
                  transcription_errors=doc_errors,
                  irr_ambiguous=is_ambiguous)
        pdf_count += 1
        qr_count += 1

# Capital call notice PDFs
for fund_id, calls in all_capital_calls.items():
    fund_def = fund_def_lookup[fund_id]
    generator = CAPITAL_CALL_GENERATORS[fund_def.report_quality_tier]

    for call in calls:
        filename = call.source_document_id
        output_path = os.path.join(volume_path, filename)
        doc_errors = call_transcription_errors.get(filename, None)
        generator(call, output_path, transcription_errors=doc_errors)
        pdf_count += 1
        cc_count += 1

# Distribution notice PDFs
for fund_id, dists in all_distributions.items():
    fund_def = fund_def_lookup[fund_id]
    generator = DISTRIBUTION_GENERATORS[fund_def.report_quality_tier]

    for dist in dists:
        filename = dist.source_document_id
        output_path = os.path.join(volume_path, filename)
        doc_errors = dist_transcription_errors.get(filename, None)
        generator(dist, output_path, transcription_errors=doc_errors)
        pdf_count += 1
        dn_count += 1

print(f"{pdf_count} PDFs generated in {volume_path}")
print(f"  Quarterly reports: {qr_count}")
print(f"  Capital calls:     {cc_count}")
print(f"  Distributions:     {dn_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Write Ground Truth to Delta Tables

# COMMAND ----------

from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType
)

# Build ground truth rows for all document types
ground_truth_rows = []
gt_pc_rows = []
gt_cc_rows = []
gt_dn_rows = []
manifest_rows = []

# Quarterly reports
for fund_id, snapshots in all_snapshots.items():
    fund_def = fund_def_lookup[fund_id]

    for snap in snapshots:
        if snap.actual_or_estimated == "not_yet_active":
            continue

        ground_truth_rows.append({
            "document_type": "quarterly_report",
            "fund_id": snap.fund_id,
            "fund_name": snap.fund_name,
            "gp_name": snap.gp_name,
            "reporting_period": snap.reporting_period,
            "quarter_end_date": snap.quarter_end_date,
            "report_date": snap.report_date,
            "source_document_id": snap.source_document_id,
            "actual_or_estimated": snap.actual_or_estimated,
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
                "document_type": "quarterly_report",
                "fund_id": snap.fund_id,
                "fund_name": snap.fund_name,
                "reporting_period": snap.reporting_period,
                "report_quality_tier": snap.report_quality_tier,
                "source_document_id": snap.source_document_id,
                "path": os.path.join(volume_path, snap.source_document_id),
            })

# Capital calls
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

# Distributions
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

# Add restated quarterly reports
for snap in restated_snapshots:
    ground_truth_rows.append({
        "document_type": "quarterly_report",
        "fund_id": snap.fund_id,
        "fund_name": snap.fund_name,
        "gp_name": snap.gp_name,
        "reporting_period": snap.reporting_period,
        "quarter_end_date": snap.quarter_end_date,
        "report_date": snap.report_date,
        "source_document_id": snap.source_document_id,
        "actual_or_estimated": snap.actual_or_estimated,
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
print(f"  Quarterly reports: {len([r for r in ground_truth_rows if r.get('document_type') == 'quarterly_report'])}")
print(f"  Capital calls:     {len(gt_cc_rows)}")
print(f"  Distributions:     {len(gt_dn_rows)}")
print(f"Portfolio company rows: {len(gt_pc_rows)}")
print(f"Manifest entries: {len(manifest_rows)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Delta Tables

# COMMAND ----------

# Ground truth uses a flexible schema — different document types have different fields.
# We use the superset approach: all columns present, NULLs for fields that don't apply.
gt_df = spark.createDataFrame(ground_truth_rows)
gt_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth")
print(f"Wrote {gt_df.count()} rows to {catalog}.{schema}.ground_truth")

# COMMAND ----------

pc_schema = StructType([
    StructField("fund_id", StringType()),
    StructField("reporting_period", StringType()),
    StructField("actual_or_estimated", StringType()),
    StructField("company_name", StringType()),
    StructField("sector", StringType()),
    StructField("investment_date", StringType()),
    StructField("initial_cost_mm", FloatType()),
    StructField("fair_value_mm", FloatType()),
])

pc_df = spark.createDataFrame(gt_pc_rows, schema=pc_schema)
pc_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth_portfolio_companies")
print(f"Wrote {pc_df.count()} rows to {catalog}.{schema}.ground_truth_portfolio_companies")

# COMMAND ----------

# Capital call ground truth (separate table for typed queries)
cc_df = spark.createDataFrame(gt_cc_rows)
cc_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth_capital_calls")
print(f"Wrote {cc_df.count()} rows to {catalog}.{schema}.ground_truth_capital_calls")

# COMMAND ----------

# Distribution ground truth (separate table for typed queries)
dn_df = spark.createDataFrame(gt_dn_rows)
dn_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.ground_truth_distributions")
print(f"Wrote {dn_df.count()} rows to {catalog}.{schema}.ground_truth_distributions")

# COMMAND ----------

manifest_schema = StructType([
    StructField("document_type", StringType()),
    StructField("fund_id", StringType()),
    StructField("fund_name", StringType()),
    StructField("reporting_period", StringType()),
    StructField("report_quality_tier", StringType()),
    StructField("source_document_id", StringType()),
    StructField("path", StringType()),
])

manifest_df = spark.createDataFrame(manifest_rows, schema=manifest_schema)
manifest_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.report_manifest")
manifest_total = manifest_df.count()
qr_manifest = manifest_df.filter("document_type = 'quarterly_report'").count()
cc_manifest = manifest_df.filter("document_type = 'capital_call'").count()
dn_manifest = manifest_df.filter("document_type = 'distribution'").count()
print(f"Wrote {manifest_total} rows to {catalog}.{schema}.report_manifest")
print(f"  Quarterly reports: {qr_manifest}")
print(f"  Capital calls:     {cc_manifest}")
print(f"  Distributions:     {dn_manifest}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Flags
# MAGIC
# MAGIC Transcription errors and IRR ambiguity flags — these are PDF-only artefacts (ground truth stays correct).
# MAGIC Written to a table so the Lakeview dashboard can query them via SQL.

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

for doc_id, errs in call_transcription_errors.items():
    flag_rows.append({
        "source_document_id": doc_id,
        "flag_type": "Transcription Error",
        "details": f"Capital call amount shown as ${errs['call_amount_mm']:,.1f}mm",
    })

for doc_id, errs in dist_transcription_errors.items():
    if "distribution_type" in errs:
        detail = f"Distribution type shown as {errs['distribution_type']}"
    else:
        detail = f"Distribution amount shown as ${errs['distribution_amount_mm']:,.1f}mm"
    flag_rows.append({
        "source_document_id": doc_id,
        "flag_type": "Transcription Error",
        "details": detail,
    })

flag_schema = StructType([
    StructField("source_document_id", StringType()),
    StructField("flag_type", StringType()),
    StructField("details", StringType()),
])

flag_df = spark.createDataFrame(flag_rows, schema=flag_schema)
flag_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.data_quality_flags")
print(f"Wrote {flag_df.count()} rows to {catalog}.{schema}.data_quality_flags")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Preview

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ground Truth Sample

# COMMAND ----------

display(
    spark.table(f"{catalog}.{schema}.ground_truth")
    .select("fund_id", "reporting_period", "actual_or_estimated", "nav_mm", "tvpi", "dpi", "rvpi", "currency", "report_quality_tier")
    .orderBy("fund_id", "reporting_period")
    .limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Freshness Analysis (Q3 2025)

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT
            actual_or_estimated,
            COUNT(*) as fund_count,
            ROUND(SUM(nav_mm), 1) as total_nav_mm
        FROM {catalog}.{schema}.ground_truth
        WHERE reporting_period = 'Q3 2025'
        GROUP BY actual_or_estimated
        ORDER BY actual_or_estimated
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quality Tier Distribution

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT
            report_quality_tier,
            COUNT(*) as report_count,
            ROUND(SUM(nav_mm), 1) as total_nav_mm
        FROM {catalog}.{schema}.ground_truth
        WHERE actual_or_estimated = 'actual'
        GROUP BY report_quality_tier
        ORDER BY report_quality_tier
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary

# COMMAND ----------

print(f"{'='*60}")
print(f"GENERATION COMPLETE")
print(f"{'='*60}")
print(f"  PDFs generated:       {pdf_count}")
print(f"    - Quarterly reports: {qr_count}")
print(f"    - Capital calls:     {cc_count}")
print(f"    - Distributions:     {dn_count}")
print(f"  Ground truth rows:    {len(ground_truth_rows)}")
print(f"    - actual:           {actual_reports}")
print(f"    - carry_forward:    {carry_forward}")
print(f"    - restated:         {len(restated_snapshots)}")
print(f"    - capital calls:    {len(gt_cc_rows)}")
print(f"    - distributions:    {len(gt_dn_rows)}")
print(f"  Portfolio companies:  {len(gt_pc_rows)}")
print(f"  Volume path:          {volume_path}")
print(f"{'='*60}")
