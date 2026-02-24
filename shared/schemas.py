"""
Shared extraction schemas, routing thresholds, and field maps.

Single Source of Truth for all document-type definitions used by:
- notebooks/03_calibration_and_routing.py
- notebooks/fast_rebuild.py
- local/03_calibration_and_routing.py
"""

# ---------------------------------------------------------------------------
# Extraction schemas — field name -> field type
# ---------------------------------------------------------------------------

QUARTERLY_REPORT_SCHEMA = {
    "fund_name": "string",
    "gp_name": "string",
    "reporting_period": "string",
    "quarter_end_date": "string",
    "vintage_year": "integer",
    "strategy": "string",
    "currency": "string",
    "committed_capital_mm": "number",
    "called_capital_mm": "number",
    "distributed_capital_mm": "number",
    "nav_mm": "number",
    "net_irr_pct": "number",
    "tvpi": "number",
    "dpi": "number",
    "management_fee_mm": "number",
    "other_expenses_mm": "number",
    "num_portfolio_companies": "integer",
}

CAPITAL_CALL_SCHEMA = {
    "fund_name": "string",
    "gp_name": "string",
    "call_date": "string",
    "due_date": "string",
    "call_amount_mm": "number",
    "call_amount_pct": "number",
    "cumulative_called_mm": "number",
    "unfunded_commitment_mm": "number",
    "bank_name": "string",
    "account_name": "string",
    "account_number": "string",
    "routing_number": "string",
    "swift_code": "string",
    "iban": "string",
    "lp_commitment_reference": "string",
    "vintage_year": "integer",
    "currency": "string",
}

DISTRIBUTION_SCHEMA = {
    "fund_name": "string",
    "gp_name": "string",
    "distribution_date": "string",
    "distribution_amount_mm": "number",
    "distribution_type": "string",
    "cumulative_distributed_mm": "number",
    "realization_source": "string",
    "lp_commitment_reference": "string",
    "vintage_year": "integer",
    "currency": "string",
}

# Schema lookup by document type
SCHEMAS = {
    "quarterly_report": QUARTERLY_REPORT_SCHEMA,
    "capital_call": CAPITAL_CALL_SCHEMA,
    "distribution": DISTRIBUTION_SCHEMA,
}

# ---------------------------------------------------------------------------
# Ground truth field name mappings (extraction field -> ground truth field)
# ---------------------------------------------------------------------------

GT_FIELD_MAPS = {
    "quarterly_report": {},
    "capital_call": {},
    "distribution": {},
}

# ---------------------------------------------------------------------------
# Document-type-specific routing thresholds
# Distribution type classification errors have tax implications — route more conservatively
# ---------------------------------------------------------------------------

ROUTING_THRESHOLDS = {
    "quarterly_report": {
        "high_confidence": 0.85,
        "medium_confidence": 0.65,
        "low_confidence": 0.40,
    },
    "capital_call": {
        "high_confidence": 0.80,
        "medium_confidence": 0.60,
        "low_confidence": 0.35,
    },
    "distribution": {
        "high_confidence": 0.90,
        "medium_confidence": 0.75,
        "low_confidence": 0.50,
    },
}
