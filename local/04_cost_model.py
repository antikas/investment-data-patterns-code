"""
04 - Cost Model: Rules-Based vs LLM Extraction (Local) - Multi-Document Type Support

Cost model updated to include ALL document types in volume calculations.
Capital calls and distributions are typically simpler documents but higher volume.
Shows how the economics change when the LLM handles all three document types 
with the same infrastructure.

KEY UPDATES:
- Volume calculations include quarterly reports, capital calls, AND distributions
- Document-type-specific volume patterns (capital calls > distributions > quarterly reports)
- Shows crossover point shift when handling all document types vs quarterly reports only
- Economics comparison: same LLM infrastructure handles 3x+ the volume

Usage:
    python local/04_cost_model.py

Requires:
    pip install numpy matplotlib
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "local" / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Cost model parameters with multi-document-type support
# ---------------------------------------------------------------------------

# Rules-based (per document layout/type)
RULES_FIXED_COST = 80_000          # Initial pipeline build
RULES_PER_LAYOUT_COST = 3_000      # Per new document layout (per document type per counterparty)
RULES_ANNUAL_MAINT_PER_LAYOUT = 1_200  # Annual maintenance per layout

# LLM-based (shared infrastructure across document types)
LLM_FIXED_COST = 150_000           # Pipeline + confidence infrastructure (handles ALL doc types)
LLM_PER_COUNTERPARTY_COST = 500    # Per-counterparty onboarding (all document types)
LLM_ANNUAL_MAINTENANCE = 40_000    # Annual platform maintenance
LLM_API_COST_PER_DOC = 0.50        # Per extraction API call (any document type)

# Document volume assumptions per counterparty per year
QUARTERLY_REPORTS_PER_YEAR = 4     # Q1, Q2, Q3, Q4
CAPITAL_CALLS_PER_YEAR = 6         # More frequent, capital deployment phase
DISTRIBUTIONS_PER_YEAR = 3         # Less frequent, later in fund lifecycle

TOTAL_DOCS_PER_COUNTERPARTY_YEAR = (
    QUARTERLY_REPORTS_PER_YEAR + 
    CAPITAL_CALLS_PER_YEAR + 
    DISTRIBUTIONS_PER_YEAR
)  # 13 documents per counterparty per year

# Document type complexity (rules-based layouts needed)
# Each document type may have multiple layouts per counterparty
LAYOUTS_PER_COUNTERPARTY = {
    "quarterly_reports": 1,        # Standardized quarterly report format
    "capital_calls": 1.5,          # Some variations in call notice formats  
    "distributions": 1.2,          # Some variations in distribution formats
}

TOTAL_LAYOUTS_PER_COUNTERPARTY = sum(LAYOUTS_PER_COUNTERPARTY.values())  # 3.7 layouts

MAX_COUNTERPARTIES = 500


def save_chart(fig, name: str):
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"  Chart saved: {path}")


def main():
    counterparties = np.arange(1, MAX_COUNTERPARTIES + 1)

    # ------------------------------------------------------------------
    # Year 1 TCO - Multi-Document Type Support
    # ------------------------------------------------------------------
    print("Cost Model: Rules-Based vs LLM Extraction - Multi-Document Type Support")
    print("=" * 80)
    print(f"Document volume per counterparty per year:")
    print(f"  Quarterly reports: {QUARTERLY_REPORTS_PER_YEAR}")
    print(f"  Capital calls:     {CAPITAL_CALLS_PER_YEAR}") 
    print(f"  Distributions:     {DISTRIBUTIONS_PER_YEAR}")
    print(f"  Total documents:   {TOTAL_DOCS_PER_COUNTERPARTY_YEAR}")
    print(f"\nRules-based complexity:")
    print(f"  Layouts per counterparty: {TOTAL_LAYOUTS_PER_COUNTERPARTY:.1f}")
    print(f"  (Different formats for each document type)")
    print(f"\nLLM infrastructure:")
    print(f"  Shared extraction pipeline handles all document types")
    print(f"  Same confidence/calibration/routing infrastructure")

    # Rules-based: cost scales with layouts (document types × counterparties)
    total_layouts = TOTAL_LAYOUTS_PER_COUNTERPARTY * counterparties
    rules_year1 = (RULES_FIXED_COST
                   + (RULES_PER_LAYOUT_COST * total_layouts)
                   + (RULES_ANNUAL_MAINT_PER_LAYOUT * total_layouts))

    # LLM-based: cost scales with counterparties + total documents
    total_docs = TOTAL_DOCS_PER_COUNTERPARTY_YEAR * counterparties
    llm_year1 = (LLM_FIXED_COST
                 + (LLM_PER_COUNTERPARTY_COST * counterparties)
                 + LLM_ANNUAL_MAINTENANCE
                 + (LLM_API_COST_PER_DOC * total_docs))

    # Crossover analysis
    crossover_idx = np.argmax(rules_year1 > llm_year1)
    crossover_n = counterparties[crossover_idx] if rules_year1[crossover_idx] > llm_year1[crossover_idx] else None

    print(f"\n{'='*60}")
    print(f"YEAR 1 COST ANALYSIS")
    print(f"{'='*60}")
    
    if crossover_n:
        print(f"Multi-document crossover: {crossover_n} counterparties")
        print(f"  Below {crossover_n}: rules-based is cheaper")
        print(f"  Above {crossover_n}: LLM + confidence infra is cheaper")
        print(f"  Cost at crossover: ${rules_year1[crossover_idx]:,.0f}")
        print(f"  Documents processed at crossover: {TOTAL_DOCS_PER_COUNTERPARTY_YEAR * crossover_n:,}")
    else:
        print("No crossover in modelled range - LLM is always more expensive")
        
    # Compare with quarterly-reports-only scenario
    qr_only_docs = QUARTERLY_REPORTS_PER_YEAR * counterparties
    qr_only_llm_cost = (LLM_FIXED_COST + (LLM_PER_COUNTERPARTY_COST * counterparties) + 
                        LLM_ANNUAL_MAINTENANCE + (LLM_API_COST_PER_DOC * qr_only_docs))
    
    qr_only_crossover_idx = np.argmax(rules_year1 > qr_only_llm_cost)
    qr_only_crossover = counterparties[qr_only_crossover_idx] if rules_year1[qr_only_crossover_idx] > qr_only_llm_cost[qr_only_crossover_idx] else None
    
    if qr_only_crossover:
        print(f"\nQuarterly-reports-only crossover: {qr_only_crossover} counterparties")
        print(f"  (For comparison: handling only quarterly reports)")
        if crossover_n:
            print(f"  Multi-document type advantage: {qr_only_crossover - crossover_n} fewer counterparties to break even")
            volume_advantage = ((TOTAL_DOCS_PER_COUNTERPARTY_YEAR * crossover_n) / 
                              (QUARTERLY_REPORTS_PER_YEAR * qr_only_crossover))
            print(f"  Volume efficiency: {volume_advantage:.1f}x more documents processed at crossover")

    # Chart: Year 1 with document type comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(counterparties, rules_year1 / 1000, color="#dc2626", linewidth=2.5, 
           label="Rules-based (all document types)")
    ax.plot(counterparties, llm_year1 / 1000, color="#2563eb", linewidth=2.5, 
           label="LLM - Multi-document (QR + CC + DN)")
    ax.plot(counterparties, qr_only_llm_cost / 1000, color="#7c3aed", linewidth=2, 
           linestyle="--", alpha=0.8, label="LLM - Quarterly reports only")

    if crossover_n:
        crossover_cost = rules_year1[crossover_idx] / 1000
        ax.axvline(x=crossover_n, color="#94a3b8", linestyle="--", alpha=0.7)
        ax.annotate(
            f"Multi-document crossover\n{crossover_n} counterparties\n(${crossover_cost:.0f}K)",
            xy=(crossover_n, crossover_cost),
            xytext=(crossover_n + MAX_COUNTERPARTIES * 0.08, crossover_cost * 1.15),
            arrowprops=dict(arrowstyle="->", color="#64748b"),
            fontsize=10, color="#334155",
        )
        
    if qr_only_crossover:
        qr_crossover_cost = rules_year1[qr_only_crossover_idx] / 1000
        ax.axvline(x=qr_only_crossover, color="#a855f7", linestyle=":", alpha=0.7)
        ax.annotate(
            f"QR-only crossover\n{qr_only_crossover} counterparties",
            xy=(qr_only_crossover, qr_crossover_cost),
            xytext=(qr_only_crossover - MAX_COUNTERPARTIES * 0.1, qr_crossover_cost * 1.3),
            arrowprops=dict(arrowstyle="->", color="#a855f7"),
            fontsize=10, color="#7c2d92",
        )

    ax.set_xlabel("Number of Counterparties", fontsize=12)
    ax.set_ylabel("Total Cost of Ownership ($K)", fontsize=12)
    ax.set_title("Year 1 TCO: Multi-Document Type Impact\n(Quarterly Reports + Capital Calls + Distributions)", 
                fontsize=14, fontweight="bold", pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:.0f}K"))

    save_chart(fig, "cost_model_multi_document_year1")

    # ------------------------------------------------------------------
    # 3-Year TCO with 10% Annual Counterparty Growth
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"3-YEAR TCO ANALYSIS (10% annual growth)")
    print(f"{'='*60}")

    growth_rate = 0.10
    years = 3

    rules_3yr = np.zeros_like(counterparties, dtype=float)
    llm_3yr = np.zeros_like(counterparties, dtype=float)
    qr_only_3yr = np.zeros_like(counterparties, dtype=float)

    for year in range(years):
        effective_cp = counterparties * (1 + growth_rate) ** year
        effective_layouts = TOTAL_LAYOUTS_PER_COUNTERPARTY * effective_cp
        effective_docs = TOTAL_DOCS_PER_COUNTERPARTY_YEAR * effective_cp
        effective_qr_docs = QUARTERLY_REPORTS_PER_YEAR * effective_cp

        if year == 0:
            rules_3yr += (RULES_FIXED_COST
                          + (RULES_PER_LAYOUT_COST * total_layouts)
                          + (RULES_ANNUAL_MAINT_PER_LAYOUT * total_layouts))
            llm_3yr += (LLM_FIXED_COST
                        + (LLM_PER_COUNTERPARTY_COST * counterparties)
                        + LLM_ANNUAL_MAINTENANCE
                        + (LLM_API_COST_PER_DOC * total_docs))
            qr_only_3yr += (LLM_FIXED_COST
                            + (LLM_PER_COUNTERPARTY_COST * counterparties)
                            + LLM_ANNUAL_MAINTENANCE
                            + (LLM_API_COST_PER_DOC * qr_only_docs))
        else:
            new_cp = counterparties * growth_rate
            new_layouts = TOTAL_LAYOUTS_PER_COUNTERPARTY * new_cp
            rules_3yr += (RULES_PER_LAYOUT_COST * new_layouts) + (RULES_ANNUAL_MAINT_PER_LAYOUT * effective_layouts)
            llm_3yr += (LLM_PER_COUNTERPARTY_COST * new_cp) + LLM_ANNUAL_MAINTENANCE + (LLM_API_COST_PER_DOC * effective_docs)
            qr_only_3yr += (LLM_PER_COUNTERPARTY_COST * new_cp) + LLM_ANNUAL_MAINTENANCE + (LLM_API_COST_PER_DOC * effective_qr_docs)

    # 3-year crossovers
    crossover_3yr_idx = np.argmax(rules_3yr > llm_3yr)
    crossover_3yr = counterparties[crossover_3yr_idx] if rules_3yr[crossover_3yr_idx] > llm_3yr[crossover_3yr_idx] else None

    qr_crossover_3yr_idx = np.argmax(rules_3yr > qr_only_3yr)
    qr_crossover_3yr = counterparties[qr_crossover_3yr_idx] if rules_3yr[qr_crossover_3yr_idx] > qr_only_3yr[qr_crossover_3yr_idx] else None

    if crossover_3yr:
        print(f"3-year multi-document crossover: {crossover_3yr} counterparties")
        print(f"  Cost at crossover: ${rules_3yr[crossover_3yr_idx]:,.0f}")
        print(f"  (vs Year 1 crossover: {crossover_n or 'N/A'} -- growth accelerates LLM advantage)")

    if qr_crossover_3yr:
        print(f"3-year QR-only crossover: {qr_crossover_3yr} counterparties")
        if crossover_3yr:
            improvement = qr_crossover_3yr - crossover_3yr
            print(f"  Multi-document advantage: {improvement} fewer counterparties needed")

    # Chart: 3-year TCO
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(counterparties, rules_3yr / 1000, color="#dc2626", linewidth=2.5, 
           label="Rules-based (3-year TCO)")
    ax.plot(counterparties, llm_3yr / 1000, color="#2563eb", linewidth=2.5, 
           label="LLM - Multi-document (3-year TCO)")
    ax.plot(counterparties, qr_only_3yr / 1000, color="#7c3aed", linewidth=2, 
           linestyle="--", alpha=0.8, label="LLM - QR only (3-year TCO)")

    if crossover_3yr:
        crossover_cost_3yr = rules_3yr[crossover_3yr_idx] / 1000
        ax.axvline(x=crossover_3yr, color="#94a3b8", linestyle="--", alpha=0.7)
        ax.annotate(
            f"3-year crossover\n{crossover_3yr} counterparties",
            xy=(crossover_3yr, crossover_cost_3yr),
            xytext=(crossover_3yr + MAX_COUNTERPARTIES * 0.08, crossover_cost_3yr * 1.1),
            arrowprops=dict(arrowstyle="->", color="#64748b"),
            fontsize=10, color="#334155",
        )

    ax.set_xlabel("Number of Counterparties", fontsize=12)
    ax.set_ylabel("3-Year Total Cost ($K)", fontsize=12)
    ax.set_title("3-Year TCO: Document Volume Economics\n(Multi-Document vs Quarterly Reports Only)", 
                fontsize=14, fontweight="bold", pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:.0f}K"))

    save_chart(fig, "cost_model_multi_document_3year")

    # ------------------------------------------------------------------
    # Document Volume Impact Analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"DOCUMENT VOLUME IMPACT")
    print(f"{'='*60}")
    
    sample_sizes = [50, 100, 200, 300]
    print(f"{'Counterparties':<15} {'QR Only':<10} {'Multi-Doc':<12} {'Volume':<8} {'Advantage':<10}")
    print(f"{'-'*60}")
    
    for n in sample_sizes:
        if n <= MAX_COUNTERPARTIES:
            qr_docs = QUARTERLY_REPORTS_PER_YEAR * n
            multi_docs = TOTAL_DOCS_PER_COUNTERPARTY_YEAR * n
            volume_mult = multi_docs / qr_docs
            
            qr_cost = (LLM_FIXED_COST + LLM_PER_COUNTERPARTY_COST * n + 
                      LLM_ANNUAL_MAINTENANCE + LLM_API_COST_PER_DOC * qr_docs)
            multi_cost = (LLM_FIXED_COST + LLM_PER_COUNTERPARTY_COST * n + 
                         LLM_ANNUAL_MAINTENANCE + LLM_API_COST_PER_DOC * multi_docs)
            
            marginal_cost = multi_cost - qr_cost
            cost_per_extra_doc = marginal_cost / (multi_docs - qr_docs)
            
            print(f"{n:<15} {qr_docs:<10,} {multi_docs:<12,} {volume_mult:<8.1f}x ${cost_per_extra_doc:<9.2f}")

    print(f"\nKEY INSIGHTS:")
    print(f"- Same LLM infrastructure handles {TOTAL_DOCS_PER_COUNTERPARTY_YEAR/QUARTERLY_REPORTS_PER_YEAR:.1f}x more documents")
    print(f"- Marginal cost per additional document type: ~${LLM_API_COST_PER_DOC:.2f}")
    print(f"- Rules-based cost scales with document type complexity (layouts)")
    print(f"- LLM cost scales primarily with document volume, not complexity")
    
    if crossover_n and qr_only_crossover:
        efficiency_gain = qr_only_crossover - crossover_n
        print(f"- Multi-document efficiency: {efficiency_gain} fewer counterparties to break even")

    # ------------------------------------------------------------------
    # Hidden Costs: Rules Maintenance Breakdown
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"HIDDEN COSTS: RULES MAINTENANCE BREAKDOWN")
    print(f"{'='*60}")
    print(f"Rules-based maintenance isn't just 'update the rule.' It includes:")
    print(f"  - Template change detection (GP changed their report format)")
    print(f"  - Rule regression testing across all document types")
    print(f"  - Silent failure investigation (extraction produced a plausible wrong value)")
    print(f"The silent failure cost is the one most teams underestimate.\n")

    template_change_rate = 0.15   # 15% of counterparties change template per year
    silent_failure_rate = 0.05    # 5% of extractions have undetected errors
    analyst_hourly_rate = 85      # cost of analyst time to investigate
    investigation_hours = 2       # hours per silent failure investigation

    print(f"{'CPs':<6} {'Template Changes':<18} {'Routine Maint':<16} {'Silent Failures':<18} {'Total Hidden':<14}")
    print(f"{'-'*72}")

    for n in [30, 100, 200, 300]:
        docs_per_year = n * TOTAL_DOCS_PER_COUNTERPARTY_YEAR
        layouts = n * TOTAL_LAYOUTS_PER_COUNTERPARTY
        template_changes = layouts * template_change_rate
        silent_failures = docs_per_year * silent_failure_rate

        template_cost = template_changes * RULES_PER_LAYOUT_COST
        routine_cost = layouts * RULES_ANNUAL_MAINT_PER_LAYOUT
        silent_cost = silent_failures * investigation_hours * analyst_hourly_rate
        total_hidden = template_cost + routine_cost + silent_cost

        print(f"{n:<6} ${template_cost:>14,.0f}  ${routine_cost:>12,.0f}  ${silent_cost:>14,.0f}  ${total_hidden:>10,.0f}")

    # ------------------------------------------------------------------
    # Write results JSON for dashboard
    # ------------------------------------------------------------------
    import json
    sample_indices = np.concatenate([[0], np.arange(9, MAX_COUNTERPARTIES, 10)])
    sample_cp = counterparties[sample_indices]

    cost_results = {
        "parameters": {
            "rules_fixed_cost": RULES_FIXED_COST,
            "rules_per_layout_cost": RULES_PER_LAYOUT_COST,
            "rules_annual_maint_per_layout": RULES_ANNUAL_MAINT_PER_LAYOUT,
            "llm_fixed_cost": LLM_FIXED_COST,
            "llm_per_counterparty_cost": LLM_PER_COUNTERPARTY_COST,
            "llm_annual_maintenance": LLM_ANNUAL_MAINTENANCE,
            "llm_api_cost_per_doc": LLM_API_COST_PER_DOC,
            "total_docs_per_counterparty_year": TOTAL_DOCS_PER_COUNTERPARTY_YEAR,
            "layouts_per_counterparty": TOTAL_LAYOUTS_PER_COUNTERPARTY,
            "growth_rate": growth_rate,
        },
        "crossover": {
            "year1_counterparties": int(crossover_n) if crossover_n else None,
            "year1_cost": float(rules_year1[crossover_idx]) if crossover_n else None,
            "three_year_counterparties": int(crossover_3yr) if crossover_3yr else None,
            "three_year_cost": float(rules_3yr[crossover_3yr_idx]) if crossover_3yr else None,
        },
        "cost_curves": [
            {
                "counterparties": int(sample_cp[i]),
                "rules_year1": float(rules_year1[sample_indices[i]]),
                "llm_year1": float(llm_year1[sample_indices[i]]),
                "rules_3yr": float(rules_3yr[sample_indices[i]]),
                "llm_3yr": float(llm_3yr[sample_indices[i]]),
            }
            for i in range(len(sample_indices))
        ],
    }

    results_path = OUTPUT_DIR / "cost_model_results.json"
    with open(results_path, "w") as f:
        json.dump(cost_results, f, indent=2)
    print(f"\nResults written to: {results_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"COST MODEL SUMMARY - MULTI-DOCUMENT TYPE SUPPORT")
    print(f"{'='*70}")
    print(f"Document types included: quarterly reports, capital calls, distributions")
    print(f"Annual document volume per counterparty: {TOTAL_DOCS_PER_COUNTERPARTY_YEAR}")
    print(f"Rules-based layouts per counterparty: {TOTAL_LAYOUTS_PER_COUNTERPARTY:.1f}")
    print(f"")
    if crossover_n:
        print(f"Year 1 crossover (multi-document): {crossover_n} counterparties")
        print(f"  Documents at crossover: {TOTAL_DOCS_PER_COUNTERPARTY_YEAR * crossover_n:,}")
        print(f"  Cost at crossover: ${rules_year1[crossover_idx]:,.0f}")
    if crossover_3yr:
        print(f"3-year crossover (multi-document): {crossover_3yr} counterparties")
        print(f"  3-year cost at crossover: ${rules_3yr[crossover_3yr_idx]:,.0f}")
    print(f"")
    print(f"LLM advantage: shared extraction infrastructure across all document types")
    print(f"Rules disadvantage: separate layouts/maintenance for each document type")
    print(f"Charts saved: {CHARTS_DIR}/cost_model_multi_document_*.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()