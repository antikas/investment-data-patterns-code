"""
Generate synthetic GP quarterly report PDFs and ground truth JSON files.

Local version — runs without Spark/Databricks. Outputs to local/output/.
Same generation logic as the Databricks notebook, different I/O layer.

Usage:
    python local/01_generate_synthetic_reports.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path so we can import shared/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.fund_definitions import FUND_DEFINITIONS, QUARTERS, FX_RATES, FX_RATE_DATE
from shared.simulation import simulate_fund_quarters, print_portfolio_summary, generate_capital_call_events, generate_distribution_events
from shared.failure_modes import (
    apply_timing_variation,
    apply_silent_restatements,
    apply_transcription_errors,
    apply_irr_ambiguity,
)
from shared.report_generators import GENERATORS
from shared.notice_generators import CAPITAL_CALL_GENERATORS, DISTRIBUTION_GENERATORS

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "local" / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Generating synthetic GP documents (quarterly reports, capital calls, distributions)...")
    print(f"  Funds: {len(FUND_DEFINITIONS)}")
    print(f"  Quarters: {len(QUARTERS)} ({QUARTERS[0]} to {QUARTERS[-1]})")
    print(f"  Output: {OUTPUT_DIR}\n")

    # -----------------------------------------------------------------------
    # 1. Simulate all funds and generate events
    # -----------------------------------------------------------------------
    print("Step 1: Simulating fund quarters and generating events...")
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
    print(f"  {total_snaps} quarter-snapshots generated")
    print(f"  {total_calls} capital call events generated")
    print(f"  {total_dists} distribution events generated\n")

    # -----------------------------------------------------------------------
    # 2. Apply failure modes (only to quarterly reports for now)
    # -----------------------------------------------------------------------
    print("Step 2: Applying failure modes to quarterly reports...")
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
    print(f"  Actual reports: {actual_reports}")
    print(f"  Carry-forward (missing): {carry_forward}")
    print(f"  Restated rows: {len(restated_snapshots)}")
    print(f"  Transcription errors: {len(transcription_errors)} reports")
    print(f"  IRR ambiguous: {len(irr_ambiguous_docs)} reports")
    
    # Apply failure modes to capital calls and distribution notices
    # For simplicity, just apply transcription errors to some notices
    import random
    rng = random.Random(42)
    
    # Apply transcription errors to ~10% of capital calls
    call_transcription_errors = {}
    for fund_id, calls in all_capital_calls.items():
        for call in calls:
            if rng.random() < 0.1:  # 10% chance
                call_transcription_errors[call.source_document_id] = {
                    "call_amount_mm": call.call_amount_mm * rng.uniform(0.9, 1.1)
                }
    
    # Apply transcription errors to ~10% of distributions  
    dist_transcription_errors = {}
    for fund_id, dists in all_distributions.items():
        for dist in dists:
            if rng.random() < 0.1:  # 10% chance
                # Sometimes get the distribution type wrong
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
    
    print(f"  Capital call transcription errors: {len(call_transcription_errors)}")
    print(f"  Distribution transcription errors: {len(dist_transcription_errors)}\n")

    # -----------------------------------------------------------------------
    # 3. Generate PDFs
    # -----------------------------------------------------------------------
    print("Step 3: Generating PDFs...")
    fund_def_lookup = {fd.fund_id: fd for fd in FUND_DEFINITIONS}
    pdf_count = 0

    # Generate quarterly report PDFs
    print("  Generating quarterly reports...")
    for fund_id, snapshots in all_snapshots.items():
        fund_def = fund_def_lookup[fund_id]
        generator = GENERATORS[fund_def.report_quality_tier]

        for snap in snapshots:
            # Only generate PDFs for actual reports (not carry-forward or not_yet_active)
            if snap.actual_or_estimated != "actual":
                continue

            filename = snap.source_document_id
            output_path = str(REPORTS_DIR / filename)

            # Get transcription errors for this specific document
            doc_errors = transcription_errors.get(filename, None)
            is_ambiguous = filename in irr_ambiguous_docs

            generator(snap, fund_def, output_path,
                      transcription_errors=doc_errors,
                      irr_ambiguous=is_ambiguous)

            size_kb = os.path.getsize(output_path) / 1024
            markers = []
            if doc_errors:
                markers.append("ERR")
            if is_ambiguous:
                markers.append("AMB")
            marker_str = f" [{','.join(markers)}]" if markers else ""
            print(f"    [QR/{fund_def.report_quality_tier:>11}] {filename} ({size_kb:.0f} KB){marker_str}")
            pdf_count += 1
            
    # Generate capital call PDFs
    print("  Generating capital call notices...")
    for fund_id, calls in all_capital_calls.items():
        fund_def = fund_def_lookup[fund_id]
        generator = CAPITAL_CALL_GENERATORS[fund_def.report_quality_tier]
        
        for call in calls:
            filename = call.source_document_id
            output_path = str(REPORTS_DIR / filename)
            
            # Get transcription errors for this document
            doc_errors = call_transcription_errors.get(filename, None)
            
            generator(call, output_path, transcription_errors=doc_errors)
            
            size_kb = os.path.getsize(output_path) / 1024
            marker_str = " [ERR]" if doc_errors else ""
            print(f"    [CC/{fund_def.report_quality_tier:>11}] {filename} ({size_kb:.0f} KB){marker_str}")
            pdf_count += 1
            
    # Generate distribution PDFs  
    print("  Generating distribution notices...")
    for fund_id, dists in all_distributions.items():
        fund_def = fund_def_lookup[fund_id]
        generator = DISTRIBUTION_GENERATORS[fund_def.report_quality_tier]
        
        for dist in dists:
            filename = dist.source_document_id
            output_path = str(REPORTS_DIR / filename)
            
            # Get transcription errors for this document
            doc_errors = dist_transcription_errors.get(filename, None)
            
            generator(dist, output_path, transcription_errors=doc_errors)
            
            size_kb = os.path.getsize(output_path) / 1024
            marker_str = " [ERR]" if doc_errors else ""
            print(f"    [DN/{fund_def.report_quality_tier:>11}] {filename} ({size_kb:.0f} KB){marker_str}")
            pdf_count += 1

    print(f"\n  {pdf_count} PDFs generated in {REPORTS_DIR}\n")

    # -----------------------------------------------------------------------
    # 4. Build ground truth JSON
    # -----------------------------------------------------------------------
    print("Step 4: Writing ground truth...")

    ground_truth = []
    gt_portfolio_companies = []
    gt_capital_calls = []
    gt_distributions = []
    manifest = []

    # Quarterly reports
    for fund_id, snapshots in all_snapshots.items():
        fund_def = fund_def_lookup[fund_id]

        for snap in snapshots:
            # Skip not-yet-active quarters
            if snap.actual_or_estimated == "not_yet_active":
                continue

            gt_row = {
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
                "fx_rate_to_usd": snap.fx_rate_to_usd,
                "fx_rate_date": snap.fx_rate_date,
                "committed_capital_mm": snap.committed_capital_mm,
                "called_capital_mm": snap.called_capital_mm,
                "distributed_capital_mm": snap.distributed_capital_mm,
                "nav_mm": snap.nav_mm,
                "net_irr_pct": snap.net_irr_pct,
                "gross_irr_pct": snap.gross_irr_pct,
                "tvpi": snap.tvpi,
                "dpi": snap.dpi,
                "rvpi": snap.rvpi,
                "management_fee_mm": snap.management_fee_mm,
                "carried_interest_mm": snap.carried_interest_mm,
                "other_expenses_mm": snap.other_expenses_mm,
                "report_quality_tier": snap.report_quality_tier,
                "num_portfolio_companies": len(snap.portfolio_companies),
            }
            ground_truth.append(gt_row)

            # Portfolio company detail
            for pc in snap.portfolio_companies:
                gt_portfolio_companies.append({
                    "fund_id": snap.fund_id,
                    "reporting_period": snap.reporting_period,
                    "actual_or_estimated": snap.actual_or_estimated,
                    "company_name": pc.name,
                    "sector": pc.sector,
                    "investment_date": pc.investment_date,
                    "initial_cost_mm": pc.initial_cost_mm,
                    "fair_value_mm": pc.fair_value_mm,
                })

            # Manifest (only actual reports with PDFs)
            if snap.actual_or_estimated == "actual" and snap.source_document_id:
                manifest.append({
                    "document_type": "quarterly_report",
                    "fund_id": snap.fund_id,
                    "fund_name": snap.fund_name,
                    "reporting_period": snap.reporting_period,
                    "report_quality_tier": snap.report_quality_tier,
                    "source_document_id": snap.source_document_id,
                    "path": str(REPORTS_DIR / snap.source_document_id),
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
                "committed_capital_mm": call.committed_capital_mm,
                "lp_commitment_mm": call.lp_commitment_mm,
                "call_date": call.call_date,
                "due_date": call.due_date,
                "call_amount_mm": call.call_amount_mm,
                "call_amount_pct": call.call_amount_pct,
                "cumulative_called_mm": call.cumulative_called_mm,
                "unfunded_commitment_mm": call.unfunded_commitment_mm,
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
            gt_capital_calls.append(gt_row)
            
            # Add to main ground truth and manifest
            ground_truth.append(gt_row)
            manifest.append({
                "document_type": "capital_call",
                "fund_id": call.fund_id,
                "fund_name": call.fund_name,
                "call_date": call.call_date,
                "report_quality_tier": call.report_quality_tier,
                "source_document_id": call.source_document_id,
                "path": str(REPORTS_DIR / call.source_document_id),
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
                "committed_capital_mm": dist.committed_capital_mm,
                "lp_commitment_mm": dist.lp_commitment_mm,
                "distribution_date": dist.distribution_date,
                "distribution_amount_mm": dist.distribution_amount_mm,
                "distribution_type": dist.distribution_type,
                "cumulative_distributed_mm": dist.cumulative_distributed_mm,
                "realization_source": dist.realization_source,
                "lp_commitment_reference": dist.lp_commitment_reference,
                "report_quality_tier": dist.report_quality_tier,
                "source_document_id": dist.source_document_id,
                "terminology": dist.terminology,
            }
            gt_distributions.append(gt_row)
            
            # Add to main ground truth and manifest
            ground_truth.append(gt_row)
            manifest.append({
                "document_type": "distribution",
                "fund_id": dist.fund_id,
                "fund_name": dist.fund_name,
                "distribution_date": dist.distribution_date,
                "report_quality_tier": dist.report_quality_tier,
                "source_document_id": dist.source_document_id,
                "path": str(REPORTS_DIR / dist.source_document_id),
            })

    # Add restated quarterly reports to ground truth
    for snap in restated_snapshots:
        gt_row = {
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
            "fx_rate_to_usd": snap.fx_rate_to_usd,
            "fx_rate_date": snap.fx_rate_date,
            "committed_capital_mm": snap.committed_capital_mm,
            "called_capital_mm": snap.called_capital_mm,
            "distributed_capital_mm": snap.distributed_capital_mm,
            "nav_mm": snap.nav_mm,
            "net_irr_pct": snap.net_irr_pct,
            "gross_irr_pct": snap.gross_irr_pct,
            "tvpi": snap.tvpi,
            "dpi": snap.dpi,
            "rvpi": snap.rvpi,
            "management_fee_mm": snap.management_fee_mm,
            "carried_interest_mm": snap.carried_interest_mm,
            "other_expenses_mm": snap.other_expenses_mm,
            "report_quality_tier": snap.report_quality_tier,
            "num_portfolio_companies": len(snap.portfolio_companies),
        }
        ground_truth.append(gt_row)

    # Write JSON files
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    pc_path = OUTPUT_DIR / "ground_truth_portfolio_companies.json"
    with open(pc_path, "w") as f:
        json.dump(gt_portfolio_companies, f, indent=2)

    calls_path = OUTPUT_DIR / "ground_truth_capital_calls.json"
    with open(calls_path, "w") as f:
        json.dump(gt_capital_calls, f, indent=2)
        
    dists_path = OUTPUT_DIR / "ground_truth_distributions.json"
    with open(dists_path, "w") as f:
        json.dump(gt_distributions, f, indent=2)

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  ground_truth.json: {len(ground_truth)} rows (all document types)")
    print(f"  ground_truth_portfolio_companies.json: {len(gt_portfolio_companies)} rows")
    print(f"  ground_truth_capital_calls.json: {len(gt_capital_calls)} rows")
    print(f"  ground_truth_distributions.json: {len(gt_distributions)} rows")
    print(f"  manifest.json: {len(manifest)} entries")

    # -----------------------------------------------------------------------
    # 5. Portfolio validation (benchmark comparison)
    # -----------------------------------------------------------------------
    print_portfolio_summary(all_snapshots)

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  PDFs generated:       {pdf_count}")
    print(f"    - Quarterly reports: {actual_reports}")
    print(f"    - Capital calls:     {total_calls}")
    print(f"    - Distributions:     {total_dists}")
    print(f"  Ground truth rows:    {len(ground_truth)}")
    print(f"    - actual reports:    {actual_reports}")
    print(f"    - carry_forward:     {carry_forward}")
    print(f"    - restated:          {len(restated_snapshots)}")
    print(f"    - capital calls:     {len(gt_capital_calls)}")
    print(f"    - distributions:     {len(gt_distributions)}")
    print(f"  Portfolio companies:  {len(gt_portfolio_companies)}")
    print(f"  Output directory:     {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
