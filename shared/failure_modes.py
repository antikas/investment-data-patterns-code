"""
GP report failure mode injectors.

Implements the 7 failure modes that real LP data teams encounter:
1. Format variation — handled by 3 quality tiers (not in this file)
2. Timing variation — missing quarters (carry-forward)
3. Terminology variation — handled by terminology sets (not in this file)
4. Currency variation — handled by fund definitions (not in this file)
5. Silent restatements — prior-quarter NAV quietly revised
6. Transcription errors — TVPI/sum inconsistencies in PDF text
7. IRR ambiguity — gross/net unlabelled

This file handles #2, #5, #6, #7 via post-processing of simulated snapshots.
"""

import random
from copy import deepcopy
from typing import Dict, List, Set, Tuple

from shared.fund_definitions import FundDefinition
from shared.simulation import QuarterSnapshot


# ---------------------------------------------------------------------------
# 2. Timing variation — missing quarters
# ---------------------------------------------------------------------------

# Funds that miss quarters (fund_id -> list of quarters to skip)
MISSING_QUARTERS = {
    "VCA-2020-I": ["Q2 2024", "Q1 2025", "Q3 2025"],  # poor tier, 75-day reporting, 3 missing
    "REA-2020-I": ["Q3 2024", "Q3 2025"],               # poor tier, 90-day reporting, 2 missing
    "GEQ-2021-II": ["Q4 2024", "Q2 2025", "Q3 2025"],   # poor tier, 90-day reporting, 3 missing
    "BYT-2022-II": ["Q1 2025"],                           # poor tier, 1 missing
    "VCA-2023-I": ["Q3 2025"],                            # poor tier, 75-day reporting, latest quarter stale
    "GEQ-2024-I": ["Q1 2024"],                            # poor tier, too early (fund just started)
}


def apply_timing_variation(
    all_snapshots: Dict[str, List[QuarterSnapshot]],
) -> Dict[str, List[QuarterSnapshot]]:
    """
    Mark missing quarters as carry-forward.

    For missing quarters:
    - actual_or_estimated = "carry_forward"
    - report_date = None
    - source_document_id = None
    - NAV/multiples carried forward from previous quarter
    """
    for fund_id, missing_qs in MISSING_QUARTERS.items():
        if fund_id not in all_snapshots:
            continue

        snapshots = all_snapshots[fund_id]
        prev_snap = None

        for i, snap in enumerate(snapshots):
            if snap.reporting_period in missing_qs and snap.actual_or_estimated == "actual":
                # Carry forward from previous quarter
                if prev_snap is not None:
                    snap.nav_mm = prev_snap.nav_mm
                    snap.called_capital_mm = prev_snap.called_capital_mm
                    snap.distributed_capital_mm = prev_snap.distributed_capital_mm
                    snap.tvpi = prev_snap.tvpi
                    snap.dpi = prev_snap.dpi
                    snap.rvpi = prev_snap.rvpi
                    snap.net_irr_pct = prev_snap.net_irr_pct
                    snap.gross_irr_pct = prev_snap.gross_irr_pct
                    snap.management_fee_mm = 0.0
                    snap.carried_interest_mm = 0.0
                    snap.other_expenses_mm = 0.0
                    snap.portfolio_companies = deepcopy(prev_snap.portfolio_companies)

                snap.actual_or_estimated = "carry_forward"
                snap.report_date = None
                snap.source_document_id = None

            if snap.actual_or_estimated in ("actual", "not_yet_active"):
                prev_snap = snap

    return all_snapshots


# ---------------------------------------------------------------------------
# 5. Silent restatements
# ---------------------------------------------------------------------------

# Funds with silent restatements: (fund_id, quarter_restated, nav_adjustment_pct)
# The restatement appears in the NEXT quarter's report for the prior period
RESTATEMENTS = [
    ("BYT-2018-I", "Q2 2024", 0.025),     # institutional, 2.5% upward restatement
    ("VCA-2019-I", "Q4 2024", -0.018),     # narrative, 1.8% downward restatement
    ("INF-2020-I", "Q1 2025", 0.013),      # institutional, 1.3% upward restatement
]


def apply_silent_restatements(
    all_snapshots: Dict[str, List[QuarterSnapshot]],
) -> Tuple[Dict[str, List[QuarterSnapshot]], List[QuarterSnapshot]]:
    """
    Create restated rows for funds with silent restatements.

    Returns:
    - Updated snapshots (original rows unchanged)
    - List of restated snapshots (new rows with actual_or_estimated="restated")

    The restated row has:
    - Same fund_id and reporting_period as the original
    - Different NAV (adjusted by the restatement percentage)
    - Recalculated TVPI/RVPI
    - actual_or_estimated = "restated"
    - source_document_id = None (appears in next quarter's report, not standalone)
    """
    restated_snapshots = []

    for fund_id, quarter, adj_pct in RESTATEMENTS:
        if fund_id not in all_snapshots:
            continue

        for snap in all_snapshots[fund_id]:
            if snap.reporting_period == quarter and snap.actual_or_estimated == "actual":
                restated = deepcopy(snap)
                restated.actual_or_estimated = "restated"
                restated.source_document_id = None

                # Adjust NAV
                restated.nav_mm = round(snap.nav_mm * (1.0 + adj_pct), 1)

                # Recalculate multiples
                if restated.called_capital_mm > 0:
                    restated.tvpi = round(
                        (restated.nav_mm + restated.distributed_capital_mm) / restated.called_capital_mm, 2
                    )
                    restated.rvpi = round(restated.tvpi - restated.dpi, 2)

                restated_snapshots.append(restated)
                break

    return all_snapshots, restated_snapshots


# ---------------------------------------------------------------------------
# 6. Transcription errors (PDF text only — ground truth stays correct)
# ---------------------------------------------------------------------------

# Returns dict mapping source_document_id -> {field: wrong_value}
# PDF generators use wrong values; ground truth retains correct values.

def apply_transcription_errors(
    all_snapshots: Dict[str, List[QuarterSnapshot]],
) -> Dict[str, Dict[str, float]]:
    """
    Select poor-tier reports for transcription errors.

    Error types:
    - TVPI off by 0.01-0.02x
    - Portfolio company fair values that don't sum correctly

    Returns: {source_document_id: {field: wrong_value}}
    Only affects PDF rendering, NOT ground truth.
    """
    rng = random.Random("transcription_errors_seed")
    errors = {}

    # Select specific poor-tier fund+quarter combos
    error_targets = [
        ("VCA-2020-I", "Q3 2024"),
        ("REA-2020-I", "Q1 2025"),
        ("BYT-2022-II", "Q3 2025"),
        ("GEQ-2024-I", "Q3 2025"),
    ]

    for fund_id, quarter in error_targets:
        if fund_id not in all_snapshots:
            continue

        for snap in all_snapshots[fund_id]:
            if snap.reporting_period != quarter or snap.actual_or_estimated != "actual":
                continue
            if snap.source_document_id is None:
                continue

            doc_id = snap.source_document_id
            errors[doc_id] = {}

            # TVPI error: off by 0.01 or 0.02
            tvpi_error = rng.choice([0.01, 0.02, -0.01, -0.02])
            errors[doc_id]["tvpi"] = round(snap.tvpi + tvpi_error, 2)

            # Portfolio company sum error: one company's value off by $1-3mm
            if snap.portfolio_companies:
                pc_idx = rng.randint(0, len(snap.portfolio_companies) - 1)
                pc = snap.portfolio_companies[pc_idx]
                fv_error = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, -1.0, -1.5, -2.0])
                errors[doc_id][f"pc_fair_value_{pc.name}"] = round(pc.fair_value_mm + fv_error, 1)

            break

    return errors


# ---------------------------------------------------------------------------
# 7. IRR ambiguity
# ---------------------------------------------------------------------------

def apply_irr_ambiguity(
    all_snapshots: Dict[str, List[QuarterSnapshot]],
) -> Set[str]:
    """
    Select narrative/poor-tier reports where IRR is shown ambiguously.

    Returns: set of source_document_ids where generators should show IRR
    without clearly labelling gross vs net.
    """
    ambiguous_docs = set()

    # Narrative tier: show "returns of X% / Y%" without labelling
    ambiguous_targets = [
        ("PCR-2019-I", "Q2 2025"),
        ("BYT-2019-I", "Q4 2024"),
        ("REA-2021-I", "Q3 2025"),
    ]

    for fund_id, quarter in ambiguous_targets:
        if fund_id not in all_snapshots:
            continue

        for snap in all_snapshots[fund_id]:
            if snap.reporting_period == quarter and snap.source_document_id:
                ambiguous_docs.add(snap.source_document_id)
                break

    return ambiguous_docs
