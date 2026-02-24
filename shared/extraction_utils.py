"""
Shared utilities for reconstructing nested extraction dicts from flat Delta table rows.

Notebook 02 writes flat rows (one per field per document) to the `extractions` table.
Notebooks 03 and 05 need nested dicts (one per document with {"extraction": {"field": {...}}}).
This module provides the single reconstruction implementation (SSOT).
"""

from collections import defaultdict


def reconstruct_extractions(flat_rows: list[dict]) -> list[dict]:
    """Reconstruct nested extraction dicts from flat rows.

    Args:
        flat_rows: List of dicts from the extractions table (one row per field).
            Expected keys: source_file, fund_id, report_quality_tier, document_type,
            field_name, extracted_value, raw_confidence, source_text, source_page, field_type.

    Returns:
        List of document dicts with nested extraction structure:
        {
            "fund_id": str,
            "fund_name": str | None,
            "report_quality_tier": str,
            "document_type": str,
            "source_file": str,
            "reporting_period": str | None,  # for quarterly_report
            "call_date": str | None,          # for capital_call
            "distribution_date": str | None,  # for distribution
            "extraction": {
                "field_name": {
                    "value": any,
                    "confidence": float,
                    "source_text": str | None,
                    "source_page": int | None,
                    "source_file": str | None,
                }
            }
        }
    """
    doc_groups = defaultdict(lambda: {"extraction": {}})

    for row in flat_rows:
        doc_key = row["source_file"]
        doc = doc_groups[doc_key]

        # Carry forward document-level fields from first row
        if "fund_id" not in doc:
            doc["fund_id"] = row["fund_id"]
            doc["fund_name"] = row.get("fund_name")
            doc["report_quality_tier"] = row["report_quality_tier"]
            doc["document_type"] = row.get("document_type", "quarterly_report")
            doc["source_file"] = row["source_file"]
            doc["reporting_period"] = row.get("reporting_period")

        # Parse extracted_value back to native type
        raw_val = row.get("extracted_value")
        field_type = row.get("field_type", "string")

        if raw_val is not None and field_type == "number":
            try:
                raw_val = float(raw_val)
            except (ValueError, TypeError):
                pass
        elif raw_val is not None and field_type == "integer":
            try:
                raw_val = int(float(raw_val))
            except (ValueError, TypeError):
                pass

        field_name = row["field_name"]
        doc["extraction"][field_name] = {
            "value": raw_val,
            "confidence": row.get("raw_confidence", 0.0),
            "source_text": row.get("source_text"),
            "source_page": row.get("source_page"),
            "source_file": row.get("source_file"),
        }

    # Pull key fields from extracted values for GT lookup
    for doc in doc_groups.values():
        ext = doc["extraction"]
        dt = doc.get("document_type", "quarterly_report")

        if dt == "quarterly_report" and "reporting_period" in ext:
            doc["reporting_period"] = ext["reporting_period"].get("value")
        elif dt == "capital_call" and "call_date" in ext:
            doc["call_date"] = ext["call_date"].get("value")
        elif dt == "distribution" and "distribution_date" in ext:
            doc["distribution_date"] = ext["distribution_date"].get("value")

        # Pull fund_name from extracted fields if not set
        if not doc.get("fund_name") and "fund_name" in ext:
            doc["fund_name"] = ext["fund_name"].get("value")

    return list(doc_groups.values())
