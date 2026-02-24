# Databricks notebook source
"""
03 — Calibration & Routing (Notebook) — Multi-Document Type Support

Compares extraction results against ground truth for ALL document types,
calibrates confidence scores using isotonic regression per document type AND
per field type, and simulates routing decisions with type-specific thresholds.

KEY UPDATES:
- Per-document-type calibration (capital calls vs quarterly reports vs distributions)
- Per-field-type calibration (amounts vs dates vs text fields)
- Document-type-specific routing thresholds (tax implications for distributions)
- Calibration curves shown per document type
- Routing decisions include document type context

This is the Databricks notebook version. For local execution, use local/03_calibration_and_routing.py

Usage in Databricks:
    %run ./notebooks/03_calibration_and_routing
"""

# COMMAND ----------
# MAGIC %md
# MAGIC # 03 — Calibration & Routing — Multi-Document Type Support
# MAGIC 
# MAGIC Updates from quarterly-report-only to **multi-document-type support**:
# MAGIC 
# MAGIC - **Per-document-type calibration**: capital calls have different accuracy profiles than quarterly reports
# MAGIC - **Per-field-type calibration**: amounts vs dates vs text fields
# MAGIC - **Document-type-specific routing thresholds**: distribution type errors have tax implications → route more conservatively
# MAGIC - **Calibration curves per document type**: separate reliability analysis
# MAGIC - **Routing decisions include document type**: output shows document type context

# COMMAND ----------

# Create widgets with defaults (idempotent — preserves user's current value)
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schema", "extraction_demo", "Schema Name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# Initialize Spark
spark = SparkSession.builder.appName("CalibrationRouting").getOrCreate()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Multi-Document-Type Schemas

# COMMAND ----------

# Import extraction schemas from shared SSOT
from shared.schemas import SCHEMAS, ROUTING_THRESHOLDS, GT_FIELD_MAPS

# COMMAND ----------
# MAGIC %md
# MAGIC ## Document-Type-Specific Routing Thresholds

# COMMAND ----------

# ROUTING_THRESHOLDS imported from shared.schemas (SSOT)
print("Routing thresholds by document type:")
for doc_type, thresholds in ROUTING_THRESHOLDS.items():
    print(f"  {doc_type}:")
    for level, threshold in thresholds.items():
        print(f"    {level}: {threshold}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Load ground truth and extraction results from Delta tables
ground_truth_df = spark.table(f"{catalog}.{schema}.ground_truth")
extractions_flat_df = spark.table(f"{catalog}.{schema}.extractions")

# Convert to pandas for analysis
ground_truth = ground_truth_df.toPandas().to_dict("records")
extractions_flat = extractions_flat_df.toPandas().to_dict("records")

# Reconstruct nested extraction dicts from flat rows (shared utility — SSOT)
from shared.extraction_utils import reconstruct_extractions
extractions = reconstruct_extractions(extractions_flat)
print(f"Reconstructed {len(extractions)} document-level extractions from {len(extractions_flat)} field rows")

# Index ground truth by document type and key
gt_index_qr = {}  # (fund_id, reporting_period) 
gt_index_cc = {}  # (fund_id, call_date)
gt_index_dn = {}  # (fund_id, distribution_date)

for row in ground_truth:
    doc_type = row.get("document_type", "quarterly_report")
    
    if doc_type == "quarterly_report" and row.get("actual_or_estimated") == "actual":
        key = (row["fund_id"], row["reporting_period"])
        gt_index_qr[key] = row
    elif doc_type == "capital_call":
        key = (row["fund_id"], row["call_date"])
        gt_index_cc[key] = row  
    elif doc_type == "distribution":
        key = (row["fund_id"], row["distribution_date"])
        gt_index_dn[key] = row

gt_indices = {
    "quarterly_report": gt_index_qr,
    "capital_call": gt_index_cc,
    "distribution": gt_index_dn,
}

print(f"Ground truth loaded:")
print(f"  Quarterly reports: {len(gt_index_qr)} actual fund-quarter rows")
print(f"  Capital calls: {len(gt_index_cc)} rows")
print(f"  Distributions: {len(gt_index_dn)} rows")
print(f"Extractions: {len(extractions)} documents")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Accuracy Measurement Per Document Type

# COMMAND ----------

def check_accuracy(extracted_value, true_value, field_type: str) -> bool:
    """Compare extracted value against ground truth."""
    NUMERIC_TOLERANCE = 0.01  # 1% relative tolerance
    
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

def get_routing_decision(calibrated_confidence: float, doc_type: str) -> str:
    """Determine routing decision based on calibrated confidence and document type."""
    thresholds = ROUTING_THRESHOLDS[doc_type]
    
    if calibrated_confidence >= thresholds["high_confidence"]:
        return "auto_accept"
    elif calibrated_confidence >= thresholds["medium_confidence"]:
        return "senior_review" 
    elif calibrated_confidence >= thresholds["low_confidence"]:
        return "expert_review"
    else:
        return "reject"

# Build accuracy records per document type
accuracy_records = []

for ext in extractions:
    fund_id = ext["fund_id"]
    doc_type = ext.get("document_type", "quarterly_report")
    tier = ext["report_quality_tier"]
    
    # Build key based on document type
    if doc_type == "quarterly_report":
        key = (fund_id, ext["reporting_period"])
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
    gt_field_map = GT_FIELD_MAPS.get(doc_type, {})

    for field_name, result in ext["extraction"].items():
        if field_name not in field_schema:
            continue

        field_type = field_schema[field_name]
        extracted_value = result.get("value")
        raw_confidence = result.get("confidence", 0.0)

        # Map field name to ground truth column
        gt_field = gt_field_map.get(field_name, field_name)
        true_value = gt_row.get(gt_field)

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

print(f"Accuracy records generated: {len(accuracy_records)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Accuracy Analysis by Document Type

# COMMAND ----------

if accuracy_records:
    correct_count = sum(1 for r in accuracy_records if r["is_correct"])
    total = len(accuracy_records)
    print(f"Total field comparisons: {total}")
    print(f"Overall accuracy: {correct_count/total:.1%}")

    # Accuracy by document type
    print(f"\n{'Document Type':<20} {'Fields':>7} {'Correct':>8} {'Accuracy':>9}")
    print(f"{'-'*46}")
    
    by_doc_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in accuracy_records:
        by_doc_type[r["document_type"]]["total"] += 1
        if r["is_correct"]:
            by_doc_type[r["document_type"]]["correct"] += 1
    
    for doc_type in ["quarterly_report", "capital_call", "distribution"]:
        if doc_type in by_doc_type:
            t = by_doc_type[doc_type]
            print(f"{doc_type:<20} {t['total']:>7} {t['correct']:>8} {t['correct']/t['total']:>9.1%}")

    # Accuracy by tier within each document type
    print(f"\nAccuracy by Quality Tier within Document Type:")
    for doc_type in ["quarterly_report", "capital_call", "distribution"]:
        doc_records = [r for r in accuracy_records if r["document_type"] == doc_type]
        if not doc_records:
            continue
            
        print(f"\n{doc_type.replace('_', ' ').title()}:")
        print(f"  {'Tier':<15} {'Fields':>7} {'Correct':>8} {'Accuracy':>9}")
        print(f"  {'-'*42}")
        
        by_tier = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in doc_records:
            by_tier[r["report_quality_tier"]]["total"] += 1
            if r["is_correct"]:
                by_tier[r["report_quality_tier"]]["correct"] += 1
        
        for tier in ["institutional", "narrative", "poor"]:
            if tier in by_tier:
                t = by_tier[tier]
                print(f"  {tier:<15} {t['total']:>7} {t['correct']:>8} {t['correct']/t['total']:>9.1%}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Per-Document-Type Calibration

# COMMAND ----------

def reliability_score(confidences, outcomes) -> float:
    """Calculate reliability (smaller is better) — mean absolute difference between confidence and observed frequency."""
    if len(confidences) == 0:
        return float("inf")
        
    # Bin confidences and calculate observed frequencies
    bins = np.linspace(0, 1, 11)  # 10 bins
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    reliability = 0.0
    total_samples = 0
    
    for i in range(len(bins) - 1):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) == 0:
            continue
            
        bin_confidences = np.array(confidences)[bin_mask]
        bin_outcomes = np.array(outcomes)[bin_mask]
        
        mean_confidence = np.mean(bin_confidences)
        observed_frequency = np.mean(bin_outcomes)
        bin_size = len(bin_confidences)
        
        reliability += bin_size * abs(mean_confidence - observed_frequency)
        total_samples += bin_size
    
    return reliability / max(total_samples, 1)

# Calibrate per document type
calibration_results = {}
calibrated_records = []

print("Calibrating confidence scores per document type...")

for doc_type in ["quarterly_report", "capital_call", "distribution"]:
    doc_records = [r for r in accuracy_records if r["document_type"] == doc_type]
    if len(doc_records) < 10:  # Need minimum samples for calibration
        print(f"Skipping {doc_type}: insufficient samples ({len(doc_records)})")
        continue
        
    print(f"\nCalibrating {doc_type} ({len(doc_records)} samples)...")
    
    confidences = [r["raw_confidence"] for r in doc_records]
    outcomes = [1 if r["is_correct"] else 0 for r in doc_records]
    
    # Isotonic regression calibration 
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(confidences, outcomes)
    
    # Apply calibration
    calibrated_confidences = calibrator.predict(confidences)
    
    # Update records with calibrated confidence
    for i, record in enumerate(doc_records):
        record["calibrated_confidence"] = float(calibrated_confidences[i])
        record["routing_decision"] = get_routing_decision(calibrated_confidences[i], doc_type)
        calibrated_records.append(record)
    
    # Reliability metrics
    pre_reliability = reliability_score(confidences, outcomes)
    post_reliability = reliability_score(calibrated_confidences, outcomes)
    
    calibration_results[doc_type] = {
        "samples": len(doc_records),
        "pre_calibration_reliability": pre_reliability,
        "post_calibration_reliability": post_reliability,
        "improvement": post_reliability - pre_reliability
    }
    
    print(f"  Pre-calibration reliability: {pre_reliability:.3f}")
    print(f"  Post-calibration reliability: {post_reliability:.3f}")
    print(f"  Improvement: {post_reliability - pre_reliability:+.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Calibration Visualization

# COMMAND ----------

# Generate calibration curves per document type
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confidence Calibration by Document Type", fontsize=14, fontweight="bold")

for i, doc_type in enumerate(["quarterly_report", "capital_call", "distribution"]):
    ax = axes[i]
    doc_records = [r for r in calibrated_records if r["document_type"] == doc_type]
    
    if len(doc_records) < 10:
        ax.text(0.5, 0.5, f"Insufficient data\n({len(doc_records)} samples)", 
               ha="center", va="center", transform=ax.transAxes)
        ax.set_title(doc_type.replace("_", " ").title())
        continue
        
    confidences = [r["raw_confidence"] for r in doc_records]
    outcomes = [1 if r["is_correct"] else 0 for r in doc_records]
    
    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(outcomes, confidences, n_bins=10)
    
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.plot(mean_pred, fraction_pos, "o-", linewidth=2, label="Raw confidence")
    
    # Post-calibration
    cal_confidences = [r["calibrated_confidence"] for r in doc_records]
    fraction_pos_cal, mean_pred_cal = calibration_curve(outcomes, cal_confidences, n_bins=10)
    ax.plot(mean_pred_cal, fraction_pos_cal, "s-", linewidth=2, alpha=0.8, label="Calibrated")
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(doc_type.replace("_", " ").title())
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()
display(fig)
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Routing Analysis Per Document Type

# COMMAND ----------

routing_analysis = {}

print(f"Routing analysis per document type...")
print(f"{'Document Type':<20} {'Auto':>8} {'Senior':>8} {'Expert':>8} {'Reject':>8}")
print(f"{'-'*58}")

for doc_type in ["quarterly_report", "capital_call", "distribution"]:
    doc_records = [r for r in calibrated_records if r["document_type"] == doc_type]
    if not doc_records:
        continue
        
    routing_counts = defaultdict(int)
    for r in doc_records:
        routing_counts[r["routing_decision"]] += 1
        
    total_docs = len(doc_records)
    auto_pct = routing_counts["auto_accept"] / total_docs * 100
    senior_pct = routing_counts["senior_review"] / total_docs * 100  
    expert_pct = routing_counts["expert_review"] / total_docs * 100
    reject_pct = routing_counts["reject"] / total_docs * 100
    
    print(f"{doc_type:<20} {auto_pct:>7.1f}% {senior_pct:>7.1f}% {expert_pct:>7.1f}% {reject_pct:>7.1f}%")
    
    routing_analysis[doc_type] = {
        "total_extractions": total_docs,
        "auto_accept": routing_counts["auto_accept"],
        "senior_review": routing_counts["senior_review"], 
        "expert_review": routing_counts["expert_review"],
        "reject": routing_counts["reject"],
        "auto_accept_pct": auto_pct,
        "senior_review_pct": senior_pct,
        "expert_review_pct": expert_pct,
        "reject_pct": reject_pct,
    }

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------

results = {
    "summary": {
        "total_extractions": len(calibrated_records),
        "total_accuracy": correct_count / total if accuracy_records else 0,
        "document_type_breakdown": {
            doc_type: {
                "extractions": len([r for r in calibrated_records if r["document_type"] == doc_type]),
                "accuracy": sum(1 for r in calibrated_records if r["document_type"] == doc_type and r["is_correct"]) / max(1, len([r for r in calibrated_records if r["document_type"] == doc_type]))
            }
            for doc_type in ["quarterly_report", "capital_call", "distribution"]
        }
    },
    "calibration_by_document_type": calibration_results,
    "routing_thresholds": ROUTING_THRESHOLDS,
    "routing_analysis_by_document_type": routing_analysis,
    "detailed_records": calibrated_records,
}

# -- extraction_results: field-level accuracy + routing (used by Lakeview dashboard) --
# Let PySpark infer schema from records — avoids field-name mismatches
calibrated_df = spark.createDataFrame(calibrated_records)
calibrated_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.calibration_results")
calibrated_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.extraction_results")
print(f"Wrote {calibrated_df.count()} records to calibration_results + extraction_results")

# -- calibration_results_json: full results for notebook 05 dashboard --
import json as _json
results_json = _json.dumps(results, default=str)
results_json_df = spark.createDataFrame([{"results_json": results_json}])
results_json_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.calibration_results_json")
print(f"Wrote full calibration results JSON to calibration_results_json")

# -- extraction_summary: per-tier stats for Lakeview dashboard --
summary_rows = []
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
    summary_rows.append({
        "report_quality_tier": tier,
        "total_fields": n_total,
        "overall_accuracy": n_correct / n_total if n_total else 0.0,
        "auto_accept_pct": n_auto / n_total if n_total else 0.0,
        "auto_accept_accuracy": n_auto_correct / n_auto if n_auto else 0.0,
        "routing_threshold": weighted_threshold,
    })

summary_df = spark.createDataFrame(summary_rows)
summary_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.extraction_summary")
print(f"Wrote {len(summary_rows)} rows to extraction_summary")

# -- routing_thresholds: sweep data for Lakeview threshold trade-off chart --
import numpy as np
threshold_rows = []
for t in np.arange(0.50, 1.00, 0.05):
    auto = [r for r in calibrated_records if r["calibrated_confidence"] >= t]
    review = [r for r in calibrated_records if r["calibrated_confidence"] < t]
    n_auto = len(auto)
    n_auto_correct = sum(1 for r in auto if r["is_correct"])
    n_errors_passed = n_auto - n_auto_correct
    threshold_rows.append({
        "threshold": round(float(t), 2),
        "auto_accept_accuracy": n_auto_correct / n_auto if n_auto else 0.0,
        "human_review_pct": len(review) / len(calibrated_records) if calibrated_records else 0.0,
        "errors_passed": n_errors_passed,
    })

threshold_df = spark.createDataFrame(threshold_rows)
threshold_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.routing_thresholds")
print(f"Wrote {len(threshold_rows)} rows to routing_thresholds")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("="*70)
print("CALIBRATION & ROUTING COMPLETE — MULTI-DOCUMENT-TYPE SUPPORT")
print("="*70)
print(f"Total extractions calibrated: {len(calibrated_records)}")
print(f"Document types supported: quarterly reports, capital calls, distributions")
print(f"Per-type calibration: confidence scores adjusted by document type accuracy profiles")
print(f"Per-type routing thresholds: distributions route most conservatively (tax implications)")
print(f"Key insights:")

for doc_type, analysis in routing_analysis.items():
    print(f"  {doc_type}:")
    print(f"    Auto-accept rate: {analysis['auto_accept_pct']:.1f}%")
    print(f"    Requires review: {analysis['senior_review_pct'] + analysis['expert_review_pct']:.1f}%")

print("="*70)