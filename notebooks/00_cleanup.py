# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Cleanup
# MAGIC
# MAGIC Idempotent reset script. Drops all Delta tables and deletes all PDFs from Volume.
# MAGIC Safe to run at any point in the pipeline — uses `IF EXISTS` on all drops.
# MAGIC
# MAGIC Does NOT drop the schema or catalog themselves — just empties them.

# COMMAND ----------

# Create widgets with defaults (idempotent — preserves user's current value)
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schema", "extraction_demo", "Schema Name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop Tables

# COMMAND ----------

tables = [
    # Pattern 1: Extraction Pipeline
    "ground_truth",
    "ground_truth_portfolio_companies",
    "ground_truth_capital_calls",
    "ground_truth_distributions",
    "report_manifest",
    "data_quality_flags",
    "extractions",
    "extraction_results",
    "extraction_summary",
    "routing_thresholds",
    "calibration_results",
    "calibration_results_json",
    "cost_model_results",
    "cost_model_summary",
]

for t in tables:
    spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{t}")
    print(f"  Dropped {catalog}.{schema}.{t}")

print(f"\n{len(tables)} tables dropped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete Volume Files

# COMMAND ----------

import os

volume_path = f"/Volumes/{catalog}/{schema}/reports"

if os.path.exists(volume_path):
    files = os.listdir(volume_path)
    for f in files:
        os.remove(os.path.join(volume_path, f))
    print(f"Deleted {len(files)} files from {volume_path}")
else:
    print(f"Volume path does not exist yet: {volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done
# MAGIC
# MAGIC All tables and files cleaned. Run `01_generate_synthetic_reports` to regenerate.
