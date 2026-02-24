"""
Quarter-by-quarter financial simulation for LP portfolio funds.

Rebuilt simulation engine with:
- Full-lifecycle simulation from vintage year (builds cumulative state before
  outputting the target quarters, so mature funds show realistic lifetime TVPIs)
- Strategy-specific return profiles with realistic dispersion (calibrated to
  Cambridge Associates / Preqin / Dimensional benchmarks)
- Vintage year correlation (2021 peak-deployment drag)
- Market events (correlated NAV shocks across strategies)
- Strategy-appropriate quarterly noise (VC lumpy, credit smooth)
- Outlier fund support (standouts and blowups)
- Lumpy vs smooth distribution patterns by strategy

All randomness is deterministic (seeded by fund_id) for reproducibility.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from shared.fund_definitions import (
    FundDefinition,
    PortfolioCompanyDef,
    QUARTER_END_DATES,
    FX_RATES,
    FX_RATE_DATE,
)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortfolioCompanySnapshot:
    name: str
    sector: str
    investment_date: str
    initial_cost_mm: float
    ownership_pct: float
    fair_value_mm: float


@dataclass
class QuarterSnapshot:
    # Identity
    fund_id: str
    fund_name: str
    gp_name: str
    vintage_year: int
    strategy: str
    currency: str
    committed_capital_mm: float
    fund_size_mm: float
    lp_commitment_mm: float
    report_quality_tier: str
    # Time
    reporting_period: str           # "Q3 2025"
    quarter_end_date: str           # "2025-09-30"
    report_date: Optional[str]      # "2025-11-15" or None for carry-forward
    # Financials
    called_capital_mm: float
    distributed_capital_mm: float
    nav_mm: float
    net_irr_pct: float
    gross_irr_pct: float
    tvpi: float
    dpi: float
    rvpi: float
    management_fee_mm: float
    carried_interest_mm: float
    other_expenses_mm: float
    # Portfolio
    portfolio_companies: List[PortfolioCompanySnapshot] = field(default_factory=list)
    # Metadata
    actual_or_estimated: str = "actual"
    source_document_id: Optional[str] = None
    fx_rate_to_usd: float = 1.0
    fx_rate_date: str = FX_RATE_DATE


# ---------------------------------------------------------------------------
# Strategy-specific parameters
# ---------------------------------------------------------------------------

# Capital call curve: logistic(age, midpoint, steepness) * max_pct
CALL_CURVE_PARAMS = {
    "Buyout":         {"midpoint": 2.5, "steepness": 1.8, "max_pct": 0.98},
    "Growth Equity":  {"midpoint": 2.8, "steepness": 1.5, "max_pct": 0.95},
    "Infrastructure": {"midpoint": 3.0, "steepness": 1.3, "max_pct": 0.92},
    "Private Credit": {"midpoint": 2.0, "steepness": 2.2, "max_pct": 0.99},
    "Secondaries":    {"midpoint": 1.0, "steepness": 3.0, "max_pct": 0.97},
    "Real Estate":    {"midpoint": 2.5, "steepness": 1.6, "max_pct": 0.95},
    "Venture Capital": {"midpoint": 2.8, "steepness": 1.4, "max_pct": 0.93},
}

# Fraction of called capital actually deployed into investments (rest = fees + cash)
DEPLOYMENT_RATIO = {
    "Buyout":         0.90,
    "Growth Equity":  0.88,
    "Infrastructure": 0.88,
    "Private Credit": 0.92,
    "Secondaries":    0.93,
    "Real Estate":    0.87,
    "Venture Capital": 0.85,
}

# ---------------------------------------------------------------------------
# Strategy + tier appreciation rates (annual NET, on deployed capital)
#
# Calibrated against published benchmarks:
# - Cambridge Associates US PE/VC (Calendar Year 2024, First Half 2025)
# - Dimensional (6,000+ funds, 1980-2022)
# - Canterbury Consulting dispersion analysis (2002-2017 vintages)
# - CAIS performance dispersion studies
#
# These are annual growth rates on deployed capital. Net IRR will be lower
# due to fee drag and cash holdback. Rates are set ~2-3% above target IRR
# to account for this.
# ---------------------------------------------------------------------------

STRATEGY_RETURN_PROFILES = {
    "Buyout": {
        "top":    (0.18, 0.26),    # top quartile: strong operational value creation
        "median": (0.11, 0.16),    # median: solid returns
        "bottom": (0.00, 0.07),    # bottom: underperforming but not catastrophic
    },
    "Growth Equity": {
        "top":    (0.20, 0.30),    # growth: higher ceiling than buyout
        "median": (0.12, 0.18),
        "bottom": (-0.04, 0.06),   # some losses in down markets
    },
    "Infrastructure": {
        "top":    (0.12, 0.17),    # infra: tight range, yield-heavy
        "median": (0.08, 0.12),
        "bottom": (0.03, 0.07),    # infra bottom still usually positive
    },
    "Private Credit": {
        "top":    (0.10, 0.13),    # credit: yield-driven, narrow dispersion
        "median": (0.07, 0.10),
        "bottom": (0.04, 0.06),    # credit almost always positive (Std Dev 9.2%)
    },
    "Secondaries": {
        "top":    (0.17, 0.24),    # discount capture drives alpha
        "median": (0.12, 0.16),
        "bottom": (0.05, 0.10),    # only 1.4% of funds below 1.0x TVPI
    },
    "Real Estate": {
        "top":    (0.15, 0.22),    # value-add top quartile
        "median": (0.09, 0.14),
        "bottom": (-0.05, 0.05),   # rate rises can hurt significantly
    },
    "Venture Capital": {
        "top":    (0.28, 0.50),    # VC top: power law, home runs
        "median": (0.10, 0.18),    # VC median: modest
        "bottom": (-0.12, 0.03),   # VC bottom: substantial losses common (Std Dev 34.2%)
    },
}

# ---------------------------------------------------------------------------
# Vintage year adjustments (added to base appreciation rate)
#
# 2021 vintages deployed at peak valuations (15-17x EBITDA for buyout,
# 100x+ revenue for VC SaaS). Carta data shows 2021 VC median TVPI ~0.95x
# at Q3 2025 vs 1.76x for 2017 at same age. Buyout effect is moderate
# (3-5 percentage points of TVPI behind 2018).
# ---------------------------------------------------------------------------

VINTAGE_ADJUSTMENTS = {
    2017: 0.02,    # mature, harvesting at good multiples
    2018: 0.01,    # solid vintage
    2019: 0.00,    # neutral baseline
    2020: 0.01,    # COVID entry = some bargain prices
    2021: -0.03,   # deployed at peak valuations — the hangover vintage
    2022: 0.00,    # rate shock, mixed entry points
    2023: 0.01,    # post-correction entry, slight discount
    2024: 0.00,    # too early to tell
}

# ---------------------------------------------------------------------------
# Market events: one-time NAV shocks applied in specific quarters
#
# Not catastrophic — just the normal ebb and flow that makes fund returns
# non-linear. Creates realistic quarter-to-quarter variation instead of
# smooth curves.
# ---------------------------------------------------------------------------

MARKET_EVENTS = {
    "Q2 2024": {
        # GP valuation mark-down cycle: delayed effect of 2022-2023 rate shock
        # finally flowing through to private marks.
        "Buyout": -0.02,
        "Growth Equity": -0.04,
        "Infrastructure": -0.01,
        "Private Credit": 0.00,
        "Secondaries": -0.01,
        "Real Estate": -0.05,
        "Venture Capital": -0.06,
    },
    "Q4 2024": {
        # Year-end marks + election resolution. Mild recovery in sentiment.
        "Buyout": 0.01,
        "Growth Equity": 0.02,
        "Infrastructure": 0.005,
        "Private Credit": 0.005,
        "Secondaries": 0.01,
        "Real Estate": 0.00,
        "Venture Capital": 0.02,
    },
}

# ---------------------------------------------------------------------------
# Strategy-specific quarterly noise (std dev of random perturbation)
#
# VC is lumpiest (individual company revaluations dominate quarterly marks).
# Credit is smoothest (yield accrues predictably).
# ---------------------------------------------------------------------------

QUARTERLY_NOISE_STD = {
    "Buyout":         0.020,
    "Growth Equity":  0.025,
    "Infrastructure": 0.012,
    "Private Credit": 0.006,
    "Secondaries":    0.015,
    "Real Estate":    0.022,
    "Venture Capital": 0.040,
}

# ---------------------------------------------------------------------------
# Distribution lumpiness (std dev of multiplicative noise on quarterly dist)
#
# Higher = more uneven quarter-to-quarter distributions.
# Buyout/VC: distributions come in bursts from individual exits.
# Credit/Infra: steady yield, predictable income.
# ---------------------------------------------------------------------------

DISTRIBUTION_NOISE_STD = {
    "Buyout":         0.30,
    "Growth Equity":  0.25,
    "Infrastructure": 0.10,
    "Private Credit": 0.08,
    "Secondaries":    0.20,
    "Real Estate":    0.15,
    "Venture Capital": 0.35,
}

# ---------------------------------------------------------------------------
# Outlier fund overrides: force specific funds into extreme performance
#
# Creates the real-world pattern where a portfolio has a standout and a
# problem child. Without these, all funds cluster around the median for
# their strategy, which looks artificial.
# ---------------------------------------------------------------------------

OUTLIER_OVERRIDES = {
    # VC standout: Lumen AI (enterprise AI) breaks out. Top-decile return.
    "VCA-2019-I": {
        "tier": "top",
        "appreciation_override": 0.35,
    },
    # Real estate value-add hit hard by rate rises and office vacancies.
    "REA-2020-I": {
        "tier": "bottom",
        "appreciation_override": -0.04,
    },
}

# Distribution onset (years from vintage) and ramp
DISTRIBUTION_PARAMS = {
    "Buyout":         {"onset": 3.5, "ramp_years": 2.0, "max_annual_pct": 0.20},
    "Growth Equity":  {"onset": 4.0, "ramp_years": 2.5, "max_annual_pct": 0.18},
    "Infrastructure": {"onset": 3.0, "ramp_years": 3.0, "max_annual_pct": 0.15},
    "Private Credit": {"onset": 1.5, "ramp_years": 1.0, "max_annual_pct": 0.25},
    "Secondaries":    {"onset": 2.0, "ramp_years": 2.0, "max_annual_pct": 0.22},
    "Real Estate":    {"onset": 3.5, "ramp_years": 2.5, "max_annual_pct": 0.16},
    "Venture Capital": {"onset": 4.5, "ramp_years": 3.0, "max_annual_pct": 0.15},
}

# Management fee rates (annual %)
MGMT_FEE_RATES = {
    "Buyout":         0.020,
    "Growth Equity":  0.020,
    "Infrastructure": 0.015,
    "Private Credit": 0.015,
    "Secondaries":    0.015,
    "Real Estate":    0.018,
    "Venture Capital": 0.025,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _logistic(age_years: float, midpoint: float, steepness: float, max_pct: float) -> float:
    """Logistic curve for cumulative capital called as fraction of committed."""
    if age_years <= 0:
        return 0.0
    raw = max_pct / (1.0 + math.exp(-steepness * (age_years - midpoint)))
    return min(max(raw, 0.0), max_pct)


def _distribution_rate(age_years: float, onset: float, ramp_years: float, max_pct: float) -> float:
    """Annualised fraction of NAV to distribute."""
    if age_years < onset:
        return 0.0
    ramp_progress = min((age_years - onset) / ramp_years, 1.0)
    return max_pct * ramp_progress


def _irr_from_tvpi_and_age(tvpi: float, age_years: float) -> float:
    """Approximate net IRR from TVPI and fund age."""
    if age_years < 0.25:
        return 0.0
    if tvpi <= 0:
        return -50.0
    try:
        raw_irr = (tvpi ** (1.0 / age_years) - 1.0) * 100.0
    except (ValueError, ZeroDivisionError, OverflowError):
        raw_irr = 0.0
    return max(min(raw_irr, 50.0), -30.0)


def _compute_report_date(quarter_end: str, speed_days: int, rng: random.Random) -> str:
    """Compute report_date = quarter_end + speed_days + jitter(+/-7)."""
    qe = datetime.strptime(quarter_end, "%Y-%m-%d")
    jitter = rng.randint(-7, 7)
    rd = qe + timedelta(days=speed_days + jitter)
    return rd.strftime("%Y-%m-%d")


def _quarter_end_date(q_label: str) -> str:
    """Compute quarter end date from label like 'Q3 2025'."""
    parts = q_label.split()
    q_num = int(parts[0][1])
    year = int(parts[1])
    if q_num == 1:
        return f"{year}-03-31"
    elif q_num == 2:
        return f"{year}-06-30"
    elif q_num == 3:
        return f"{year}-09-30"
    else:
        return f"{year}-12-31"


def _all_quarters_from(start_year: int, end_label: str) -> List[Tuple[str, str]]:
    """
    Generate all (quarter_label, quarter_end_date) from start_year Q1
    through end_label (inclusive).
    """
    end_parts = end_label.split()
    end_q = int(end_parts[0][1])
    end_y = int(end_parts[1])

    result = []
    for year in range(start_year, end_y + 1):
        for q in range(1, 5):
            if year == end_y and q > end_q:
                break
            label = f"Q{q} {year}"
            result.append((label, _quarter_end_date(label)))
    return result


def _allocate_nav_to_companies(
    fund_def: FundDefinition,
    nav: float,
    quarter_end: datetime,
    rng: random.Random,
    company_weights: dict,
) -> List[PortfolioCompanySnapshot]:
    """
    Allocate fund-level NAV across named portfolio companies.

    Each company gets a weight-proportional share of NAV. Weights evolve
    each quarter (some companies outperform, some underperform).
    """
    active = []
    for pc in fund_def.portfolio_companies:
        inv_date = datetime.strptime(pc.investment_date, "%Y-%m-%d")
        if inv_date <= quarter_end:
            active.append(pc)

    if not active or nav <= 0:
        return []

    # Evolve weights with noise (outperformers gain share, underperformers lose)
    total_weight = 0.0
    for pc in active:
        w = company_weights.get(pc.name, pc.initial_cost_mm)
        drift = rng.gauss(0, 0.03)  # +/- 3% drift in share
        w = max(w * (1.0 + drift), 0.01)
        company_weights[pc.name] = w
        total_weight += w

    # Allocate NAV proportionally
    snapshots = []
    for pc in active:
        share = company_weights[pc.name] / total_weight
        fv = round(nav * share, 1)
        snapshots.append(PortfolioCompanySnapshot(
            name=pc.name,
            sector=pc.sector,
            investment_date=pc.investment_date,
            initial_cost_mm=pc.initial_cost_mm,
            ownership_pct=pc.ownership_pct,
            fair_value_mm=fv,
        ))

    return snapshots


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def simulate_fund_quarters(
    fund_def: FundDefinition,
    quarters: List[str],
) -> List[QuarterSnapshot]:
    """
    Simulate quarter-by-quarter financials for a single fund.

    CRITICAL: simulates from the fund's vintage year through the latest
    requested quarter, building cumulative state over the full fund lifecycle.
    Only emits snapshots for the requested quarters. This means a 2017
    vintage fund reporting in Q3 2025 shows lifetime TVPI (8+ years of
    compounded growth), not just 7 quarters of growth.

    Approach:
    1. Fund gets a performance tier (top/median/bottom) from seeded RNG
    2. Base appreciation rate drawn from strategy + tier specific range
    3. Vintage year adjustment applied (2021 penalty, 2017 bonus)
    4. Simulate ALL quarters from vintage year, accumulating state
    5. Market events create correlated shocks in specific quarters
    6. Strategy-appropriate quarterly noise (VC lumpy, credit smooth)
    7. Distribution noise for lumpy vs smooth exit patterns
    8. Outlier funds get forced appreciation rates
    9. Only return snapshots for the requested quarters

    Returns one QuarterSnapshot per requested quarter.
    """
    rng = random.Random(fund_def.fund_id)
    strategy = fund_def.strategy

    # --- Determine fund performance ---
    outlier = OUTLIER_OVERRIDES.get(fund_def.fund_id)

    if outlier:
        perf_tier = outlier["tier"]
        annual_net_rate = outlier["appreciation_override"]
    else:
        tier_roll = rng.random()
        if tier_roll < 0.35:
            perf_tier = "top"
        elif tier_roll < 0.75:
            perf_tier = "median"
        else:
            perf_tier = "bottom"

        strategy_rates = STRATEGY_RETURN_PROFILES.get(
            strategy, STRATEGY_RETURN_PROFILES["Buyout"]
        )
        rate_low, rate_high = strategy_rates[perf_tier]
        annual_net_rate = rng.uniform(rate_low, rate_high)

    # Vintage year adjustment
    vintage_adj = VINTAGE_ADJUSTMENTS.get(fund_def.vintage_year, 0.0)
    annual_net_rate += vintage_adj

    # Strategy params
    call_params = CALL_CURVE_PARAMS.get(strategy, CALL_CURVE_PARAMS["Buyout"])
    dist_params = DISTRIBUTION_PARAMS.get(strategy, DISTRIBUTION_PARAMS["Buyout"])
    deploy_ratio = DEPLOYMENT_RATIO.get(strategy, 0.90)
    mgmt_fee_rate = MGMT_FEE_RATES.get(strategy, 0.020)
    noise_std = QUARTERLY_NOISE_STD.get(strategy, 0.020)
    dist_noise_std = DISTRIBUTION_NOISE_STD.get(strategy, 0.20)
    other_expense_rate = rng.uniform(0.001, 0.003)

    vintage_start = datetime(fund_def.vintage_year, 1, 1)

    # Generate ALL quarters from vintage year to latest requested quarter
    latest_quarter = quarters[-1]  # assumes chronologically sorted
    all_sim_quarters = _all_quarters_from(fund_def.vintage_year, latest_quarter)
    requested = set(quarters)

    # Cumulative state
    cum_called = 0.0
    cum_distributed = 0.0
    deployed_value = 0.0  # the growing pool of invested capital
    prev_nav = 0.0

    # Company allocation weights
    company_weights = {pc.name: pc.initial_cost_mm for pc in fund_def.portfolio_companies}

    snapshots = []

    for q_label, q_end_str in all_sim_quarters:
        quarter_end = datetime.strptime(q_end_str, "%Y-%m-%d")
        age_years = (quarter_end - vintage_start).days / 365.25

        # Fund hasn't started yet
        if age_years < 0:
            if q_label in requested:
                snapshots.append(_empty_snapshot(fund_def, q_label, q_end_str, rng))
            continue

        # --- Capital calls (cumulative) ---
        cum_call_pct = _logistic(
            age_years, call_params["midpoint"],
            call_params["steepness"], call_params["max_pct"],
        )
        target_called = fund_def.committed_capital_mm * cum_call_pct
        new_called = max(target_called, cum_called)
        quarter_call = new_called - cum_called
        cum_called = new_called

        # New deployment from this quarter's call
        new_deployment = quarter_call * deploy_ratio
        deployed_value += new_deployment

        # --- Grow deployed value ---
        quarterly_base_rate = annual_net_rate / 4.0

        # Market event shock (correlated across strategies)
        event = MARKET_EVENTS.get(q_label)
        market_shock = event.get(strategy, 0.0) if event else 0.0

        # Quarterly noise (strategy-specific volatility)
        noise = rng.gauss(0, noise_std)

        quarterly_growth = quarterly_base_rate + market_shock + noise
        deployed_value *= (1.0 + quarterly_growth)
        deployed_value = max(deployed_value, 0.0)

        # Cash reserve (non-deployed portion)
        cum_fees_and_cash = cum_called * (1.0 - deploy_ratio)
        cash_remaining = max(cum_fees_and_cash * 0.15, 0.0)

        # NAV before distributions
        nav_pre_dist = deployed_value + cash_remaining

        # --- Distributions (with lumpiness) ---
        dist_rate = _distribution_rate(
            age_years, dist_params["onset"],
            dist_params["ramp_years"], dist_params["max_annual_pct"],
        )
        base_distribution = nav_pre_dist * dist_rate / 4.0

        # Apply distribution noise (lumpy exits vs steady yield)
        if base_distribution > 0:
            dist_noise = max(rng.gauss(1.0, dist_noise_std), 0.0)
            quarterly_distribution = base_distribution * dist_noise
        else:
            quarterly_distribution = 0.0

        # Mature funds: accelerate if under-distributed
        if age_years > 7.0 and cum_called > 0:
            current_dpi = cum_distributed / cum_called
            if current_dpi < 0.8:
                quarterly_distribution *= 2.0

        # Cap at 25% of NAV per quarter
        quarterly_distribution = min(quarterly_distribution, nav_pre_dist * 0.25)
        cum_distributed += quarterly_distribution

        # Reduce deployed value by distributions (capital returned)
        deployed_value -= quarterly_distribution
        deployed_value = max(deployed_value, 0.0)

        # Final NAV
        nav = round(deployed_value + cash_remaining, 1)

        # --- Fees (informational — already baked into deployment ratio) ---
        if age_years <= 5.0:
            fee_basis = fund_def.committed_capital_mm
        else:
            fee_basis = max(prev_nav, 0.0)
        quarterly_mgmt_fee = round(fee_basis * mgmt_fee_rate / 4.0, 2)
        quarterly_other = round(fund_def.committed_capital_mm * other_expense_rate / 4.0, 2)

        # --- Carried interest ---
        total_value = nav + cum_distributed
        if cum_called > 0 and total_value > cum_called * 1.08:
            profit_above_hurdle = total_value - cum_called * 1.08
            carried_interest = round(max(profit_above_hurdle * 0.20, 0.0), 2)
        else:
            carried_interest = 0.0

        # --- Multiples (EXACT reconciliation) ---
        if cum_called > 0:
            tvpi = round((nav + cum_distributed) / cum_called, 2)
            dpi = round(cum_distributed / cum_called, 2)
            rvpi = round(tvpi - dpi, 2)
        else:
            tvpi, dpi, rvpi = 1.00, 0.00, 1.00

        # --- IRR ---
        net_irr = round(_irr_from_tvpi_and_age(tvpi, age_years), 1)
        gross_spread = rng.uniform(2.0, 4.0)
        gross_irr = round(net_irr + gross_spread, 1)

        # --- Portfolio company allocation (always run to advance RNG) ---
        pc_snapshots = _allocate_nav_to_companies(
            fund_def, nav, quarter_end, rng, company_weights,
        )

        prev_nav = nav

        # --- Only emit snapshot for requested quarters ---
        if q_label not in requested:
            continue

        # Report date
        report_date = _compute_report_date(
            q_end_str, fund_def.reporting_speed_days, rng,
        )

        # Source document ID
        quarter_tag = q_label.replace(" ", "_")
        source_doc_id = f"{fund_def.fund_id}_{quarter_tag}.pdf"

        snapshot = QuarterSnapshot(
            fund_id=fund_def.fund_id,
            fund_name=fund_def.fund_name,
            gp_name=fund_def.gp_name,
            vintage_year=fund_def.vintage_year,
            strategy=strategy,
            currency=fund_def.currency,
            committed_capital_mm=fund_def.committed_capital_mm,
            fund_size_mm=fund_def.fund_size_mm,
            lp_commitment_mm=fund_def.lp_commitment_mm,
            report_quality_tier=fund_def.report_quality_tier,
            reporting_period=q_label,
            quarter_end_date=q_end_str,
            report_date=report_date,
            called_capital_mm=round(cum_called, 1),
            distributed_capital_mm=round(cum_distributed, 1),
            nav_mm=nav,
            net_irr_pct=net_irr,
            gross_irr_pct=gross_irr,
            tvpi=tvpi,
            dpi=dpi,
            rvpi=rvpi,
            management_fee_mm=quarterly_mgmt_fee,
            carried_interest_mm=carried_interest,
            other_expenses_mm=quarterly_other,
            portfolio_companies=pc_snapshots,
            actual_or_estimated="actual",
            source_document_id=source_doc_id,
            fx_rate_to_usd=FX_RATES.get(fund_def.currency, 1.0),
            fx_rate_date=FX_RATE_DATE,
        )
        snapshots.append(snapshot)

    return snapshots


def _empty_snapshot(
    fund_def: FundDefinition,
    quarter: str,
    quarter_end_str: str,
    rng: random.Random,
) -> QuarterSnapshot:
    """Produce a snapshot for a quarter before the fund's vintage start."""
    return QuarterSnapshot(
        fund_id=fund_def.fund_id,
        fund_name=fund_def.fund_name,
        gp_name=fund_def.gp_name,
        vintage_year=fund_def.vintage_year,
        strategy=fund_def.strategy,
        currency=fund_def.currency,
        committed_capital_mm=fund_def.committed_capital_mm,
        fund_size_mm=fund_def.fund_size_mm,
        lp_commitment_mm=fund_def.lp_commitment_mm,
        report_quality_tier=fund_def.report_quality_tier,
        reporting_period=quarter,
        quarter_end_date=quarter_end_str,
        report_date=None,
        called_capital_mm=0.0,
        distributed_capital_mm=0.0,
        nav_mm=0.0,
        net_irr_pct=0.0,
        gross_irr_pct=0.0,
        tvpi=1.00,
        dpi=0.00,
        rvpi=1.00,
        management_fee_mm=0.0,
        carried_interest_mm=0.0,
        other_expenses_mm=0.0,
        portfolio_companies=[],
        actual_or_estimated="not_yet_active",
        source_document_id=None,
        fx_rate_to_usd=FX_RATES.get(fund_def.currency, 1.0),
        fx_rate_date=FX_RATE_DATE,
    )


# ---------------------------------------------------------------------------
# Portfolio validation summary
# ---------------------------------------------------------------------------

def print_portfolio_summary(
    all_snapshots: Dict[str, List[QuarterSnapshot]],
    latest_quarter: str = "Q3 2025",
) -> None:
    """
    Print summary statistics for validation against industry benchmarks.

    Run after simulation to eyeball whether the output looks credible.
    Compare against the benchmark tables in the plan.
    """
    from collections import defaultdict

    by_strategy = defaultdict(list)

    for fund_id, snapshots in all_snapshots.items():
        for snap in snapshots:
            if snap.reporting_period == latest_quarter and snap.actual_or_estimated == "actual":
                by_strategy[snap.strategy].append(snap)

    print(f"\n{'='*90}")
    print(f"PORTFOLIO SUMMARY — {latest_quarter}")
    print(f"{'='*90}")
    print(f"{'Strategy':<20} {'Funds':>5} {'Med IRR':>10} {'IRR Range':>18} "
          f"{'Med TVPI':>10} {'TVPI Range':>16} {'Med DPI':>10}")
    print(f"{'-'*90}")

    all_irrs = []
    all_tvpis = []

    for strategy in ["Buyout", "Growth Equity", "Infrastructure", "Private Credit",
                      "Secondaries", "Real Estate", "Venture Capital"]:
        snaps = by_strategy.get(strategy, [])
        if not snaps:
            continue

        irrs = sorted([s.net_irr_pct for s in snaps])
        tvpis = sorted([s.tvpi for s in snaps])
        dpis = sorted([s.dpi for s in snaps])
        all_irrs.extend(irrs)
        all_tvpis.extend(tvpis)

        n = len(irrs)
        med_irr = irrs[n // 2]
        med_tvpi = tvpis[n // 2]
        med_dpi = dpis[n // 2]
        min_irr, max_irr = irrs[0], irrs[-1]
        min_tvpi, max_tvpi = tvpis[0], tvpis[-1]

        print(f"{strategy:<20} {n:>5} {med_irr:>9.1f}% "
              f"{min_irr:>7.1f}% - {max_irr:>5.1f}% "
              f"{med_tvpi:>9.2f}x "
              f"{min_tvpi:>6.2f}x - {max_tvpi:>5.2f}x "
              f"{med_dpi:>9.2f}x")

    print(f"{'-'*90}")
    if all_irrs:
        all_irrs.sort()
        all_tvpis.sort()
        n = len(all_irrs)
        print(f"{'PORTFOLIO':<20} {n:>5} {all_irrs[n//2]:>9.1f}% "
              f"{all_irrs[0]:>7.1f}% - {all_irrs[-1]:>5.1f}% "
              f"{all_tvpis[n//2]:>9.2f}x "
              f"{all_tvpis[0]:>6.2f}x - {all_tvpis[-1]:>5.2f}x")
    print(f"{'='*90}")

    # Per-fund detail for the latest quarter
    print(f"\nPER-FUND DETAIL — {latest_quarter}")
    print(f"{'Fund ID':<14} {'Strategy':<18} {'Vint':>4} {'Age':>5} {'Tier':<8} "
          f"{'IRR':>7} {'TVPI':>7} {'DPI':>7} {'NAV($mm)':>10}")
    print(f"{'-'*90}")
    all_snaps = []
    for fund_id, snapshots in all_snapshots.items():
        for snap in snapshots:
            if snap.reporting_period == latest_quarter and snap.actual_or_estimated == "actual":
                all_snaps.append(snap)

    for s in sorted(all_snaps, key=lambda x: (x.strategy, x.vintage_year)):
        age = 2025.75 - s.vintage_year  # approximate age at Q3 2025
        outlier = OUTLIER_OVERRIDES.get(s.fund_id)
        tier_marker = "outlier" if outlier else ""
        print(f"{s.fund_id:<14} {s.strategy:<18} {s.vintage_year:>4} {age:>4.1f}y {tier_marker:<8} "
              f"{s.net_irr_pct:>6.1f}% {s.tvpi:>6.2f}x {s.dpi:>6.2f}x {s.nav_mm:>9.1f}")
    print(f"{'='*90}")

    # Validation checks
    print("\nVALIDATION CHECKS:")
    issues = []
    for strategy, snaps in by_strategy.items():
        for s in snaps:
            age = 2025.75 - s.vintage_year
            if strategy == "Private Credit" and s.net_irr_pct < -5:
                issues.append(f"  WARNING: {s.fund_id} credit fund has {s.net_irr_pct}% IRR (unusually negative)")
            if strategy == "Infrastructure" and s.net_irr_pct < -10:
                issues.append(f"  WARNING: {s.fund_id} infra fund has {s.net_irr_pct}% IRR (unusually negative)")
            if s.tvpi > 5.0:
                issues.append(f"  WARNING: {s.fund_id} has {s.tvpi}x TVPI (unrealistically high)")
            if s.tvpi < 0.3 and s.strategy != "Venture Capital":
                issues.append(f"  WARNING: {s.fund_id} has {s.tvpi}x TVPI (very low for {s.strategy})")
            if abs(s.tvpi - s.dpi - s.rvpi) > 0.02:
                issues.append(f"  ERROR: {s.fund_id} TVPI={s.tvpi} != DPI={s.dpi} + RVPI={s.rvpi}")
            # Check mature fund has reasonable TVPI
            if age > 6 and s.tvpi < 1.0 and strategy not in ("Real Estate", "Venture Capital"):
                issues.append(f"  NOTE: {s.fund_id} ({strategy}, {age:.0f}yr) has TVPI {s.tvpi}x — below water at maturity")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  All checks passed.")
    print()


# ---------------------------------------------------------------------------
# Capital call and distribution event generation
# ---------------------------------------------------------------------------

def generate_capital_call_events(
    fund_def: FundDefinition,
    quarters: List[str],
) -> List:
    """
    Generate individual capital call events from the fund simulation.
    
    Returns a list of capital call events where the fund actually called
    capital during that quarter (delta > threshold).
    """
    from shared.notice_generators import CapitalCallNotice, generate_bank_details
    
    # Get the quarterly snapshots
    snapshots = simulate_fund_quarters(fund_def, quarters)
    rng = random.Random(fund_def.fund_id + "_calls")
    
    events = []
    prev_called = 0.0
    
    for snap in snapshots:
        if snap.actual_or_estimated != "actual":
            continue
            
        # Calculate the delta (new capital called this quarter)
        call_delta = snap.called_capital_mm - prev_called
        
        # Only generate a capital call event if meaningful amount was called
        if call_delta < 0.01:  # Less than $100k threshold
            prev_called = snap.called_capital_mm
            continue
        
        # Generate call notice date (before quarter end)
        quarter_end = datetime.strptime(snap.quarter_end_date, "%Y-%m-%d")
        
        # Capital calls typically come 1-4 weeks before quarter end
        days_before_qe = rng.randint(7, 28)
        call_date = quarter_end - timedelta(days=days_before_qe)
        
        # Due date is typically 5-15 business days after call date
        due_days = rng.randint(5, 15)
        due_date = call_date + timedelta(days=due_days)
        
        # Generate bank details
        bank_details = generate_bank_details(fund_def, rng)
        
        # LP commitment reference (format varies by GP)
        lp_ref_formats = [
            f"LP-{fund_def.fund_id}-001",
            f"{fund_def.fund_id}_LP001", 
            f"REF{fund_def.fund_id}001",
            f"{fund_def.fund_name[:6].upper().replace(' ', '')}-LP-01"
        ]
        lp_ref = rng.choice(lp_ref_formats)
        
        # Percentage of total commitment this call represents
        call_pct = (call_delta / fund_def.committed_capital_mm) * 100
        
        # Unfunded commitment remaining after this call
        unfunded = fund_def.committed_capital_mm - snap.called_capital_mm
        
        # Source document ID
        call_date_str = call_date.strftime("%Y-%m-%d")
        source_doc_id = f"{fund_def.fund_id}_CALL_{call_date_str.replace('-', '')}.pdf"
        
        event = CapitalCallNotice(
            fund_id=fund_def.fund_id,
            fund_name=fund_def.fund_name,
            gp_name=fund_def.gp_name,
            vintage_year=fund_def.vintage_year,
            strategy=fund_def.strategy,
            currency=fund_def.currency,
            committed_capital_mm=fund_def.committed_capital_mm,
            lp_commitment_mm=fund_def.lp_commitment_mm,
            
            call_date=call_date.strftime("%Y-%m-%d"),
            due_date=due_date.strftime("%Y-%m-%d"),
            call_amount_mm=call_delta,
            call_amount_pct=call_pct,
            cumulative_called_mm=prev_called,  # Called before this call
            unfunded_commitment_mm=unfunded,
            
            bank_name=bank_details["bank_name"],
            account_name=bank_details["account_name"],
            account_number=bank_details["account_number"],
            routing_number=bank_details["routing_number"],
            swift_code=bank_details["swift_code"],
            iban=bank_details["iban"],
            
            lp_commitment_reference=lp_ref,
            
            report_quality_tier=fund_def.report_quality_tier,
            source_document_id=source_doc_id,
            terminology=fund_def.terminology,
        )
        
        events.append(event)
        prev_called = snap.called_capital_mm
    
    return events


def generate_distribution_events(
    fund_def: FundDefinition,
    quarters: List[str],
) -> List:
    """
    Generate individual distribution events from the fund simulation.
    
    Returns a list of distribution events where the fund actually distributed
    capital during that quarter (delta > threshold).
    """
    from shared.notice_generators import DistributionNotice
    
    # Get the quarterly snapshots  
    snapshots = simulate_fund_quarters(fund_def, quarters)
    rng = random.Random(fund_def.fund_id + "_dists")
    
    events = []
    prev_distributed = 0.0
    
    for snap in snapshots:
        if snap.actual_or_estimated != "actual":
            continue
            
        # Calculate the delta (new distributions this quarter)
        dist_delta = snap.distributed_capital_mm - prev_distributed
        
        # Only generate a distribution event if meaningful amount was distributed
        if dist_delta < 0.01:  # Less than $100k threshold
            prev_distributed = snap.distributed_capital_mm
            continue
        
        # Generate distribution date (typically at/after quarter end)
        quarter_end = datetime.strptime(snap.quarter_end_date, "%Y-%m-%d")
        
        # Distributions come 0-30 days after quarter end
        days_after_qe = rng.randint(0, 30)
        dist_date = quarter_end + timedelta(days=days_after_qe)
        
        # Distribution type determination
        # Early distributions more likely return of capital
        # Later distributions more likely gains
        age_years = (quarter_end - datetime(fund_def.vintage_year, 1, 1)).days / 365.25
        
        if age_years < 3:
            # Early: mostly return of capital
            dist_type_weights = [("return_of_capital", 0.8), ("income", 0.15), ("gain", 0.05)]
        elif age_years < 6:
            # Middle: mixed
            dist_type_weights = [("return_of_capital", 0.4), ("income", 0.25), ("gain", 0.35)]
        else:
            # Mature: mostly gains
            dist_type_weights = [("return_of_capital", 0.2), ("income", 0.15), ("gain", 0.65)]
        
        # Weighted random choice
        rand_val = rng.random()
        cumulative_weight = 0.0
        dist_type = "return_of_capital"  # default
        for dtype, weight in dist_type_weights:
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                dist_type = dtype
                break
        
        # Generate realization source for gains
        realization_source = None
        if dist_type == "gain" and snap.portfolio_companies:
            # Pick a portfolio company that might have been sold
            pc = rng.choice(snap.portfolio_companies)
            source_templates = [
                f"partial realization of {pc.name}",
                f"sale of {pc.name}",
                f"exit from {pc.name}",
                f"dividend from {pc.name}",
            ]
            realization_source = rng.choice(source_templates)
        elif dist_type == "income":
            income_sources = [
                "portfolio company dividends",
                "interest and fee income", 
                "management fee recaptures"
            ]
            realization_source = rng.choice(income_sources)
        
        # LP commitment reference (same format as capital calls)
        lp_ref_formats = [
            f"LP-{fund_def.fund_id}-001",
            f"{fund_def.fund_id}_LP001", 
            f"REF{fund_def.fund_id}001",
            f"{fund_def.fund_name[:6].upper().replace(' ', '')}-LP-01"
        ]
        lp_ref = rng.choice(lp_ref_formats)
        
        # Source document ID
        dist_date_str = dist_date.strftime("%Y-%m-%d")
        source_doc_id = f"{fund_def.fund_id}_DIST_{dist_date_str.replace('-', '')}.pdf"
        
        event = DistributionNotice(
            fund_id=fund_def.fund_id,
            fund_name=fund_def.fund_name,
            gp_name=fund_def.gp_name,
            vintage_year=fund_def.vintage_year,
            strategy=fund_def.strategy,
            currency=fund_def.currency,
            committed_capital_mm=fund_def.committed_capital_mm,
            lp_commitment_mm=fund_def.lp_commitment_mm,
            
            distribution_date=dist_date.strftime("%Y-%m-%d"),
            distribution_amount_mm=dist_delta,
            distribution_type=dist_type,
            cumulative_distributed_mm=prev_distributed,  # Distributed before this event
            
            realization_source=realization_source,
            lp_commitment_reference=lp_ref,
            
            report_quality_tier=fund_def.report_quality_tier,
            source_document_id=source_doc_id,
            terminology=fund_def.terminology,
        )
        
        events.append(event)
        prev_distributed = snap.distributed_capital_mm
    
    return events
