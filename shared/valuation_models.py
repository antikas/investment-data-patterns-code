"""
Pattern 2: Mark-to-Model Governance — Shared Valuation Logic

Single source of truth for:
  - PATTERN_2_TABLES and PATTERN_2_OUTPUT_FILES (imported by cleanup and pipeline scripts)
  - Dataclasses whose field names become table column names
  - Strategy -> method mapping and assumption generation parameters
  - Four embedded governance scenarios
  - Fair value computation, bridge decomposition, drift detection

No external dependencies. No LLM calls.
"""

import hashlib
import re
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shared.fund_definitions import FUND_DEFINITIONS, QUARTERS


# ---------------------------------------------------------------------------
# SSOT: table names and output file names
# ---------------------------------------------------------------------------

PATTERN_2_TABLES = [
    "model_holdings",
    "model_assumptions",
    "model_valuations",
    "model_bridges",
    "model_drift",
    "model_data_access",
]

PATTERN_2_OUTPUT_FILES = {
    "holdings":      "model_holdings.json",
    "assumptions":   "model_assumptions.json",
    "valuations":    "model_valuations.json",
    "bridges":       "model_bridges.json",
    "drift":         "model_drift.json",
    "data_access":   "model_data_access.json",
}


# ---------------------------------------------------------------------------
# Strategy -> method mapping
# ---------------------------------------------------------------------------

STRATEGY_METHODS: Dict[str, str] = {
    "Buyout":         "comparable_multiples",
    "Growth Equity":  "comparable_multiples",
    "Infrastructure": "dcf",
    "Real Estate":    "cap_rate",
}

MARK_TO_MODEL_STRATEGIES: set = set(STRATEGY_METHODS.keys())

QUARTER_INDEX: Dict[str, int] = {q: i for i, q in enumerate(QUARTERS)}


# ---------------------------------------------------------------------------
# Assumption data access — how the LP obtained assumption data per fund
# Maps to Article 02's "three sources": disclosed (side letter / LPAC),
# back-calculated (LP derives from FV + metric), extracted (GP narrative)
# ---------------------------------------------------------------------------

_FUND_GP_LOOKUP: Dict[str, str] = {fd.fund_id: fd.gp_name for fd in FUND_DEFINITIONS}

_DEFAULT_ASSUMPTION_SOURCE = "back_calculated"


@dataclass
class DataAccessRecord:
    """How the LP obtains valuation assumptions for a given fund.

    Written to model_data_access table — makes data provenance queryable.
    """
    fund_id:           str
    gp_name:           str
    assumption_source: str   # disclosed | back_calculated | extracted
    access_mechanism:  str


DATA_ACCESS_RECORDS: List[DataAccessRecord] = [
    # Disclosed: established relationship, LPAC seat or side letter
    DataAccessRecord("BYT-2017-I", _FUND_GP_LOOKUP["BYT-2017-I"], "disclosed", "LPAC reporting package"),
    DataAccessRecord("BYT-2018-I", _FUND_GP_LOOKUP["BYT-2018-I"], "disclosed", "LPAC reporting package"),
    DataAccessRecord("BYT-2019-I", _FUND_GP_LOOKUP["BYT-2019-I"], "disclosed", "Side letter - assumption disclosure"),
    DataAccessRecord("BYT-2020-I", _FUND_GP_LOOKUP["BYT-2020-I"], "disclosed", "LPAC reporting package"),
    DataAccessRecord("INF-2017-I", _FUND_GP_LOOKUP["INF-2017-I"], "disclosed", "Side letter - assumption disclosure"),
    DataAccessRecord("INF-2020-I", _FUND_GP_LOOKUP["INF-2020-I"], "disclosed", "LPAC reporting package"),
    DataAccessRecord("GEQ-2018-I", _FUND_GP_LOOKUP["GEQ-2018-I"], "disclosed", "Side letter - assumption disclosure"),
    DataAccessRecord("REA-2018-I", _FUND_GP_LOOKUP["REA-2018-I"], "disclosed", "LPAC reporting package"),
    # Back-calculated: LP derives implied multiple from reported FV + metric
    DataAccessRecord("BYT-2021-I", _FUND_GP_LOOKUP["BYT-2021-I"], "back_calculated", "Back-calculation from reported FV and operating metric"),
    DataAccessRecord("BYT-2022-I", _FUND_GP_LOOKUP["BYT-2022-I"], "back_calculated", "Back-calculation from reported FV and operating metric"),
    DataAccessRecord("GEQ-2021-I", _FUND_GP_LOOKUP["GEQ-2021-I"], "back_calculated", "Back-calculation from reported FV and operating metric"),
    DataAccessRecord("GEQ-2021-II", _FUND_GP_LOOKUP["GEQ-2021-II"], "back_calculated", "Back-calculation from reported FV and operating metric"),
    DataAccessRecord("REA-2020-I", _FUND_GP_LOOKUP["REA-2020-I"], "back_calculated", "Back-calculation from reported FV and operating metric"),
    DataAccessRecord("REA-2021-I", _FUND_GP_LOOKUP["REA-2021-I"], "back_calculated", "Back-calculation from reported FV and operating metric"),
    # Extracted: only narrative GP letter available
    DataAccessRecord("BYT-2022-II", _FUND_GP_LOOKUP["BYT-2022-II"], "extracted", "NLP extraction from GP narrative letter"),
    DataAccessRecord("BYT-2024-I", _FUND_GP_LOOKUP["BYT-2024-I"], "extracted", "NLP extraction from GP narrative letter"),
    DataAccessRecord("GEQ-2022-I", _FUND_GP_LOOKUP["GEQ-2022-I"], "extracted", "NLP extraction from GP narrative letter"),
    DataAccessRecord("GEQ-2024-I", _FUND_GP_LOOKUP["GEQ-2024-I"], "extracted", "NLP extraction from GP narrative letter"),
    DataAccessRecord("INF-2023-I", _FUND_GP_LOOKUP["INF-2023-I"], "extracted", "NLP extraction from GP narrative letter"),
]

# Lookup dict built from records — used by generate_valuation_history()
ASSUMPTION_DATA_ACCESS: Dict[str, str] = {r.fund_id: r.assumption_source for r in DATA_ACCESS_RECORDS}


# ---------------------------------------------------------------------------
# Assumption generation parameters (calibrated to industry benchmarks)
# ---------------------------------------------------------------------------

ASSUMPTION_CONFIG: Dict[str, Dict] = {
    "Buyout": {
        "multiple_min":           8.5,
        "multiple_max":           14.0,
        "underlying_metric_type": "ebitda_mm",
        "underlying_growth_qoq":  (0.008, 0.032),   # quarterly EBITDA growth range
        "multiple_noise_qoq":     (-0.18, 0.18),     # random quarterly multiple change
        "drift_per_quarter":      0.10,              # scenario 3: additive drift per quarter
    },
    "Growth Equity": {
        "multiple_min":           5.0,
        "multiple_max":           12.0,
        "underlying_metric_type": "revenue_mm",
        "underlying_growth_qoq":  (0.020, 0.060),   # growth companies grow faster
        "multiple_noise_qoq":     (-0.25, 0.25),
        "drift_per_quarter":      0.0,
    },
    "Infrastructure": {
        "discount_rate_min":      0.065,
        "discount_rate_max":      0.095,
        "tgr_min":                0.018,
        "tgr_max":                0.028,
        "underlying_metric_type": "fcf_mm",
        "underlying_growth_qoq":  (0.003, 0.012),   # slow, regulated
        "rate_noise_qoq":         (-0.0015, 0.0015), # ±15bps quarterly noise
    },
    "Real Estate": {
        "cap_rate_min":           0.042,
        "cap_rate_max":           0.068,
        "underlying_metric_type": "noi_mm",
        "underlying_growth_qoq":  (0.002, 0.008),   # rent escalation
        "cap_rate_noise_qoq":     (-0.0020, 0.0015), # asymmetric: rates more likely to expand
    },
}


# ---------------------------------------------------------------------------
# Comparable set identifiers (visible to analysts; drives cross-GP divergence)
# Each sector has three peer groups: conservative, standard, optimistic
# ---------------------------------------------------------------------------

_COMP_SETS: Dict[str, Tuple[str, str, str]] = {
    "Healthcare Services":      ("healthcare_svcs_mid_market_2024",  "healthcare_svcs_standard_2024",  "healthcare_svcs_large_cap_2024"),
    "Enterprise Software":      ("enterprise_sw_legacy_2024",        "enterprise_sw_standard_2024",    "enterprise_sw_saas_premium_2024"),
    "Media & Entertainment":    ("media_traditional_2024",           "media_diversified_2024",         "media_digital_premium_2024"),
    "Industrials":              ("industrials_cyclical_2024",        "industrials_diversified_2024",   "industrials_specialty_2024"),
    "Consumer Products":        ("consumer_staples_peers_2024",      "consumer_branded_2024",          "consumer_premium_2024"),
    "Supply Chain":             ("logistics_legacy_2024",            "logistics_standard_2024",        "logistics_tech_2024"),
    "Technology Services":      ("tech_svcs_legacy_2024",            "tech_svcs_standard_2024",        "tech_svcs_growth_2024"),
    "Financial Services":       ("fin_svcs_legacy_2024",             "fin_svcs_standard_2024",         "fin_svcs_growth_2024"),
    "Pharmaceuticals":          ("pharma_mid_cap_2024",              "pharma_standard_2024",           "pharma_growth_2024"),
    "Cloud Infrastructure":     ("cloud_infra_growth_2024",          "cloud_saas_standard_2024",       "cloud_premium_saas_2024"),
    "Fintech":                  ("fintech_lending_peers_2024",       "fintech_growth_peers_2024",      "fintech_premium_2024"),
    "Healthcare IT":            ("health_it_traditional_2024",       "health_it_standard_2024",        "health_it_growth_2024"),
    "Data & AI":                ("data_analytics_legacy_2024",       "data_ai_standard_2024",          "data_ai_premium_2024"),
    "Supply Chain & Logistics": ("logistics_tech_legacy_2024",       "logistics_tech_standard_2024",   "logistics_tech_growth_2024"),
    "Industrial Technology":    ("industrial_tech_legacy_2024",      "industrial_tech_standard_2024",  "industrial_tech_growth_2024"),
    "Automotive Parts":         ("auto_parts_cyclical_2024",         "auto_parts_standard_2024",       "auto_parts_premium_2024"),
    "Hospitality":              ("hospitality_legacy_2024",          "hospitality_standard_2024",      "hospitality_growth_2024"),
    "EdTech":                   ("edtech_legacy_2024",               "edtech_standard_2024",           "edtech_premium_2024"),
    "SaaS":                     ("saas_legacy_2024",                 "saas_standard_2024",             "saas_premium_2024"),
    "Food & Beverage":          ("food_bev_legacy_2024",             "food_bev_standard_2024",         "food_bev_growth_2024"),
    "Real Estate Services":     ("re_svcs_legacy_2024",              "re_svcs_standard_2024",          "re_svcs_growth_2024"),
    "_default":                 ("sector_peers_conservative_2024",   "sector_peers_standard_2024",     "sector_peers_growth_2024"),
}


def _comp_set(sector: str, style: str) -> str:
    """Return a comparable set identifier for the given sector and peer group style."""
    sets = _COMP_SETS.get(sector, _COMP_SETS["_default"])
    return sets[{"conservative": 0, "standard": 1, "optimistic": 2}[style]]


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

# Scenario 1: Methodology-driven swing
# BYT-2020-I (Schwarzwald Capital) expands EBITDA multiples in Q3 2024.
# Post-ECB rate cut repricing — European peer groups re-rated. Operations flat.
# Average expansion ~1.2x across 4 holdings, one position near 2x.
# Bridge: ~85% assumption-driven. Flagged for board review.
SCENARIO_SWING: Dict = {
    "fund_id":                   "BYT-2020-I",
    "trigger_quarter":           "Q3 2024",
    "base_multiple_adjustment":  0.9,    # 3 holdings get this
    "spike_multiple_adjustment": 2.0,    # 1 holding (largest) gets this
    "spike_holding_index":       0,      # first holding by generation order (Eurotech)
    "approval_status":           "flagged",
    "notes":                     "Post-ECB rate cut: European tech and industrial peer multiples expanded materially. Board review required.",
}

# Scenario 2: Hidden outperformance
# GEQ-2021-I / Helios Data Systems: revenue grows 38% but GP tightens multiple 2.5x.
# Net NAV change small. Bridge reveals masked operational strength.
SCENARIO_HIDDEN: Dict = {
    "fund_id":             "GEQ-2021-I",
    "company_name":        "Helios Data Systems",
    "trigger_quarter":     "Q2 2025",
    "metric_boost":        0.38,
    "multiple_tightening": -2.5,
    "approval_status":     "approved",
    "notes":               "Conservative peer set rotation to reflect sector multiple compression; underlying ARR growth ahead of plan.",
}

# Scenario 3: Systematic drift
# All buyout funds add +0.10x per quarter (tracking public market PE repricing).
# By Q3 2025 the average buyout multiple is ~0.60x above Q1 2024 baseline.
SCENARIO_DRIFT: Dict = {
    "strategies":        ["Buyout"],
    "drift_per_quarter": 0.10,
}

# Scenario 4: Cross-GP divergence
# BYT-2019-I (Oakmont, conservative) vs BYT-2022-I (Northstar, growth-oriented).
# Both hold Media & Entertainment assets. Different comparable sets -> different multiples.
SCENARIO_DIVERGENCE: Dict = {
    "conservative_fund":          "BYT-2019-I",
    "optimistic_fund":            "BYT-2022-I",
    "sector":                     "Media & Entertainment",
    "conservative_comp_style":    "conservative",
    "optimistic_comp_style":      "optimistic",
    "conservative_multiple_bias": -1.5,
    "optimistic_multiple_bias":   +1.8,
}


# ---------------------------------------------------------------------------
# Dataclasses — field names are the column names in every output
# ---------------------------------------------------------------------------

@dataclass
class HoldingRecord:
    holding_id:       str    # fund_id + slug(company_name)
    fund_id:          str
    gp_name:          str
    company_name:     str
    sector:           str
    strategy:         str
    initial_cost_mm:  float
    investment_date:  str    # YYYY-MM-DD
    currency:         str


@dataclass
class AssumptionRecord:
    assumption_id:       str    # holding_id + quarter slug
    holding_id:          str
    fund_id:             str
    quarter:             str
    method:              str    # comparable_multiples | dcf | cap_rate
    # comparable_multiples (Buyout uses ebitda, Growth Equity uses revenue)
    ebitda_multiple:     Optional[float]  = None
    revenue_multiple:    Optional[float]  = None
    comparable_set_id:   Optional[str]    = None
    # dcf
    discount_rate:       Optional[float]  = None
    terminal_growth_rate: Optional[float] = None
    # cap_rate
    cap_rate:            Optional[float]  = None
    # governance
    approval_status:     str              = "approved"
    notes:               Optional[str]    = None
    assumption_changed:  bool             = False
    # provenance: how LP obtained the assumption data
    assumption_source:   str              = "back_calculated"  # disclosed | back_calculated | extracted


@dataclass
class ValuationRecord:
    valuation_id:            str    # val_ + holding_id + quarter slug
    holding_id:              str
    fund_id:                 str
    quarter:                 str
    assumption_id:           str
    underlying_metric_mm:    float
    underlying_metric_type:  str    # ebitda_mm | revenue_mm | fcf_mm | noi_mm
    fair_value_mm:           float
    currency:                str


@dataclass
class BridgeRecord:
    holding_id:                   str
    fund_id:                      str
    company_name:                 str
    sector:                       str
    quarter:                      str    # quarter being bridged TO
    prev_quarter:                 str    # quarter being bridged FROM
    opening_fair_value_mm:        float
    closing_fair_value_mm:        float
    total_change_mm:              float
    operational_component_mm:     float  # change from underlying metric movement
    assumption_component_mm:      float  # change from assumption update
    methodology_component_mm:     float  # change from method switch
    prev_method:                  str
    curr_method:                  str
    assumption_change_description: str


@dataclass
class DriftRecord:
    quarter:                       str
    avg_buyout_ebitda_multiple:    float
    median_buyout_ebitda_multiple: float
    avg_geq_revenue_multiple:      float
    median_geq_revenue_multiple:   float
    avg_infra_discount_rate:       float
    median_infra_discount_rate:    float
    avg_re_cap_rate:               float
    median_re_cap_rate:            float
    total_portfolio_nav_mm:        float
    operational_change_mm:         float
    assumption_change_mm:          float
    methodology_change_mm:         float
    # NAV breakdown by assumption data source
    nav_disclosed_mm:              float = 0.0
    nav_back_calculated_mm:        float = 0.0
    nav_extracted_mm:              float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng(seed_str: str) -> random.Random:
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def _holding_id(fund_id: str, company_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", company_name.lower()).strip("_")
    return f"{fund_id}_{slug}"


def _assumption_id(holding_id: str, quarter: str) -> str:
    return f"{holding_id}_{quarter.replace(' ', '_').lower()}"


def _valuation_id(holding_id: str, quarter: str) -> str:
    return f"val_{holding_id}_{quarter.replace(' ', '_').lower()}"


def _vintage_appreciation(investment_date: str, rng: random.Random) -> float:
    """Fair-value-to-cost multiplier as of Q1 2024, based on the investment date."""
    year = int(investment_date[:4])
    if year <= 2018:
        return rng.uniform(1.45, 1.85)
    elif year <= 2020:
        return rng.uniform(1.15, 1.55)
    elif year <= 2021:
        return rng.uniform(0.95, 1.35)
    else:
        return rng.uniform(0.78, 1.05)


def _comp_style_for_holding(fund_id: str, sector: str, investment_date: str) -> str:
    """Determine comparable set style from fund identity and holding age."""
    if (fund_id == SCENARIO_DIVERGENCE["conservative_fund"]
            and sector == SCENARIO_DIVERGENCE["sector"]):
        return SCENARIO_DIVERGENCE["conservative_comp_style"]
    if (fund_id == SCENARIO_DIVERGENCE["optimistic_fund"]
            and sector == SCENARIO_DIVERGENCE["sector"]):
        return SCENARIO_DIVERGENCE["optimistic_comp_style"]
    year = int(investment_date[:4])
    if year <= 2018:
        return "conservative"
    elif year <= 2021:
        return "standard"
    return "optimistic"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_fair_value(
    underlying_metric_mm: float,
    method: str,
    ebitda_multiple:      Optional[float] = None,
    revenue_multiple:     Optional[float] = None,
    discount_rate:        Optional[float] = None,
    terminal_growth_rate: Optional[float] = None,
    cap_rate:             Optional[float] = None,
) -> float:
    """Compute fair value from underlying metric and valuation assumptions."""
    if method == "comparable_multiples":
        multiple = ebitda_multiple if ebitda_multiple is not None else revenue_multiple
        if multiple is None:
            raise ValueError("comparable_multiples requires ebitda_multiple or revenue_multiple")
        return round(underlying_metric_mm * multiple, 3)
    if method == "dcf":
        if discount_rate is None or terminal_growth_rate is None:
            raise ValueError("dcf requires discount_rate and terminal_growth_rate")
        spread = max(discount_rate - terminal_growth_rate, 0.005)
        return round(underlying_metric_mm / spread, 3)
    if method == "cap_rate":
        if cap_rate is None:
            raise ValueError("cap_rate method requires cap_rate")
        return round(underlying_metric_mm / max(cap_rate, 0.001), 3)
    raise ValueError(f"Unknown valuation method: {method}")


def compute_bridge(
    prev_val:    "ValuationRecord",
    curr_val:    "ValuationRecord",
    prev_assump: "AssumptionRecord",
    curr_assump: "AssumptionRecord",
    holding:     "HoldingRecord",
    prev_quarter: str = "",
) -> "BridgeRecord":
    """
    Decompose NAV change into three components that sum to total_change_mm:
      operational:  change driven by underlying metric (EBITDA / revenue / FCF / NOI)
      assumption:   change driven by assumption update (multiple / rate / cap rate)
      methodology:  change driven by switching valuation method
    """
    fv_open    = prev_val.fair_value_mm
    fv_close   = curr_val.fair_value_mm
    total_chg  = round(fv_close - fv_open, 3)
    metric_old = prev_val.underlying_metric_mm
    metric_new = curr_val.underlying_metric_mm
    m_old      = prev_assump.method
    m_new      = curr_assump.method

    if m_old != m_new:
        # Method switched: compute hypothetical FV under old method with new metric + old assumptions
        hypothetical = compute_fair_value(
            metric_new, m_old,
            ebitda_multiple=prev_assump.ebitda_multiple,
            revenue_multiple=prev_assump.revenue_multiple,
            discount_rate=prev_assump.discount_rate,
            terminal_growth_rate=prev_assump.terminal_growth_rate,
            cap_rate=prev_assump.cap_rate,
        )
        operational = round(hypothetical - fv_open, 3)
        methodology = round(fv_close - hypothetical, 3)
        assumption  = 0.0
        desc = f"Method change: {m_old} -> {m_new}"

    elif m_old == "comparable_multiples":
        mult_old = prev_assump.ebitda_multiple if prev_assump.ebitda_multiple is not None else prev_assump.revenue_multiple
        mult_new = curr_assump.ebitda_multiple if curr_assump.ebitda_multiple is not None else curr_assump.revenue_multiple
        operational = round((metric_new - metric_old) * mult_old, 3)
        assumption  = round(metric_new * (mult_new - mult_old), 3)
        methodology = 0.0
        label = "EBITDA" if prev_assump.ebitda_multiple is not None else "Revenue"
        desc  = f"{label} multiple: {mult_old:.2f}x -> {mult_new:.2f}x"
        if (curr_assump.comparable_set_id and prev_assump.comparable_set_id
                and curr_assump.comparable_set_id != prev_assump.comparable_set_id):
            desc += f" | comp set: {prev_assump.comparable_set_id} -> {curr_assump.comparable_set_id}"

    elif m_old == "dcf":
        spread_old  = max(prev_assump.discount_rate - prev_assump.terminal_growth_rate, 0.005)
        spread_new  = max(curr_assump.discount_rate - curr_assump.terminal_growth_rate, 0.005)
        operational = round((metric_new - metric_old) / spread_old, 3)
        assumption  = round(metric_new * (1.0 / spread_new - 1.0 / spread_old), 3)
        methodology = 0.0
        desc = (f"Discount rate: {prev_assump.discount_rate*100:.2f}%"
                f" -> {curr_assump.discount_rate*100:.2f}%")

    elif m_old == "cap_rate":
        cr_old      = max(prev_assump.cap_rate, 0.001)
        cr_new      = max(curr_assump.cap_rate, 0.001)
        operational = round((metric_new - metric_old) / cr_old, 3)
        assumption  = round(metric_new * (1.0 / cr_new - 1.0 / cr_old), 3)
        methodology = 0.0
        desc = (f"Cap rate: {prev_assump.cap_rate*100:.2f}%"
                f" -> {curr_assump.cap_rate*100:.2f}%")

    else:
        operational, assumption, methodology = total_chg, 0.0, 0.0
        desc = "Unknown method"

    # Absorb rounding residual into operational to guarantee the identity
    residual = round(total_chg - operational - assumption - methodology, 3)
    if abs(residual) < 0.05:
        operational = round(operational + residual, 3)

    return BridgeRecord(
        holding_id=holding.holding_id,
        fund_id=holding.fund_id,
        company_name=holding.company_name,
        sector=holding.sector,
        quarter=curr_val.quarter,
        prev_quarter=prev_quarter or prev_val.quarter,
        opening_fair_value_mm=fv_open,
        closing_fair_value_mm=fv_close,
        total_change_mm=total_chg,
        operational_component_mm=operational,
        assumption_component_mm=assumption,
        methodology_component_mm=methodology,
        prev_method=m_old,
        curr_method=m_new,
        assumption_change_description=desc,
    )


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_holdings() -> List[HoldingRecord]:
    """Return HoldingRecord for every portfolio company in a mark-to-model strategy."""
    holdings = []
    for fund in FUND_DEFINITIONS:
        if fund.strategy not in MARK_TO_MODEL_STRATEGIES:
            continue
        for company in fund.portfolio_companies:
            holdings.append(HoldingRecord(
                holding_id=_holding_id(fund.fund_id, company.name),
                fund_id=fund.fund_id,
                gp_name=fund.gp_name,
                company_name=company.name,
                sector=company.sector,
                strategy=fund.strategy,
                initial_cost_mm=company.initial_cost_mm,
                investment_date=company.investment_date,
                currency=fund.currency,
            ))
    return holdings


def generate_valuation_history(
    holdings: List[HoldingRecord],
) -> Tuple[List[AssumptionRecord], List[ValuationRecord]]:
    """
    Generate quarterly assumption and valuation records for all holdings
    across Q1 2024 – Q3 2025 (7 quarters).

    Four governance scenarios are embedded:
      1. Methodology swing:      BYT-2020-I, Q3 2024 — avg +1.2x multiple expansion, flagged
      2. Hidden outperformance:  GEQ-2021-I / Helios, Q2 2025 — strong metric, tightened multiple
      3. Systematic drift:       all buyout funds +0.10x/quarter (scenario 3)
      4. Cross-GP divergence:    BYT-2019-I vs BYT-2022-I, same sector, different comp sets
    """
    all_assumptions: List[AssumptionRecord] = []
    all_valuations:  List[ValuationRecord]  = []

    # Pre-compute: which holding in BYT-2020-I gets the scenario 1 spike
    _swing_fund_holdings = [h.holding_id for h in holdings
                            if h.fund_id == SCENARIO_SWING["fund_id"]]
    _swing_spike_hid = (_swing_fund_holdings[SCENARIO_SWING["spike_holding_index"]]
                        if _swing_fund_holdings else None)

    for holding in holdings:
        rng    = _rng(f"{holding.holding_id}_v2")
        cfg    = ASSUMPTION_CONFIG[holding.strategy]
        method = STRATEGY_METHODS[holding.strategy]

        # Starting fair value at Q1 2024
        appreciation = _vintage_appreciation(holding.investment_date, rng)
        initial_fv   = holding.initial_cost_mm * appreciation

        # Initialise per-method assumption state
        if method == "comparable_multiples":
            lo, hi = cfg["multiple_min"], cfg["multiple_max"]
            current_multiple = rng.uniform(lo, (lo + hi) / 2.0 + 1.0)
            current_multiple = max(lo, min(hi, current_multiple))

            # Scenario 4: structural bias by comparable set style
            if (holding.fund_id == SCENARIO_DIVERGENCE["conservative_fund"]
                    and holding.sector == SCENARIO_DIVERGENCE["sector"]):
                current_multiple += SCENARIO_DIVERGENCE["conservative_multiple_bias"]
                current_multiple = max(lo, current_multiple)
            elif (holding.fund_id == SCENARIO_DIVERGENCE["optimistic_fund"]
                    and holding.sector == SCENARIO_DIVERGENCE["sector"]):
                current_multiple += SCENARIO_DIVERGENCE["optimistic_multiple_bias"]
                current_multiple = min(hi + 1.0, current_multiple)

            comp_style    = _comp_style_for_holding(holding.fund_id, holding.sector, holding.investment_date)
            current_comp  = _comp_set(holding.sector, comp_style)
            current_metric = initial_fv / current_multiple

        elif method == "dcf":
            current_dr  = rng.uniform(cfg["discount_rate_min"], cfg["discount_rate_max"])
            current_tgr = rng.uniform(cfg["tgr_min"],           cfg["tgr_max"])
            spread       = max(current_dr - current_tgr, 0.005)
            current_metric = initial_fv * spread  # FCF = FV × spread

        else:  # cap_rate
            current_cr     = rng.uniform(cfg["cap_rate_min"], cfg["cap_rate_max"])
            current_metric = initial_fv * current_cr  # NOI = FV × cap_rate

        metric_type     = cfg["underlying_metric_type"]
        prev_assumption: Optional[AssumptionRecord] = None

        for q_idx, quarter in enumerate(QUARTERS):

            # -- Evolve underlying metric (Q1 2024 is the baseline, no change) --
            if q_idx > 0:
                lo, hi = cfg["underlying_growth_qoq"]
                growth = rng.uniform(lo, hi)

                # Scenario 2: one-quarter metric surge for Helios Data Systems
                if (holding.fund_id == SCENARIO_HIDDEN["fund_id"]
                        and holding.company_name == SCENARIO_HIDDEN["company_name"]
                        and quarter == SCENARIO_HIDDEN["trigger_quarter"]):
                    growth = SCENARIO_HIDDEN["metric_boost"]

                current_metric *= (1.0 + growth)

            # -- Evolve assumptions --
            approval_status = "approved"
            notes: Optional[str] = None
            assump_kw: dict = {}

            if method == "comparable_multiples":
                if q_idx > 0:
                    lo, hi = cfg["multiple_noise_qoq"]
                    delta = rng.uniform(lo, hi)

                    # Scenario 3: systematic buyout multiple drift
                    if holding.strategy == "Buyout":
                        delta += SCENARIO_DRIFT["drift_per_quarter"]

                    # Scenario 1: assumption swing for BYT-2020-I
                    # One holding gets the spike (~2x), others get the base (~0.9x)
                    if (holding.fund_id == SCENARIO_SWING["fund_id"]
                            and quarter == SCENARIO_SWING["trigger_quarter"]):
                        if holding.holding_id == _swing_spike_hid:
                            delta += SCENARIO_SWING["spike_multiple_adjustment"]
                        else:
                            delta += SCENARIO_SWING["base_multiple_adjustment"]
                        approval_status  = SCENARIO_SWING["approval_status"]
                        notes            = SCENARIO_SWING["notes"]

                    # Scenario 2: GP tightens multiple to offset strong metric
                    if (holding.fund_id == SCENARIO_HIDDEN["fund_id"]
                            and holding.company_name == SCENARIO_HIDDEN["company_name"]
                            and quarter == SCENARIO_HIDDEN["trigger_quarter"]):
                        delta += SCENARIO_HIDDEN["multiple_tightening"]
                        notes  = SCENARIO_HIDDEN["notes"]

                    current_multiple += delta
                    # Clamp: allow modest overshoot above range to capture scenario 1
                    current_multiple = max(cfg["multiple_min"] - 0.5,
                                          min(cfg["multiple_max"] + 3.5, current_multiple))

                if holding.strategy == "Buyout":
                    assump_kw = {
                        "ebitda_multiple":  round(current_multiple, 2),
                        "revenue_multiple": None,
                        "comparable_set_id": current_comp,
                    }
                else:  # Growth Equity
                    assump_kw = {
                        "ebitda_multiple":  None,
                        "revenue_multiple": round(current_multiple, 2),
                        "comparable_set_id": current_comp,
                    }

            elif method == "dcf":
                if q_idx > 0:
                    lo, hi = cfg["rate_noise_qoq"]
                    current_dr  = max(cfg["discount_rate_min"] - 0.010,
                                      min(cfg["discount_rate_max"] + 0.010,
                                          current_dr + rng.uniform(lo, hi)))
                    current_tgr = max(cfg["tgr_min"],
                                      min(cfg["tgr_max"],
                                          current_tgr + rng.uniform(-0.0008, 0.0008)))
                assump_kw = {
                    "discount_rate":        round(current_dr,  4),
                    "terminal_growth_rate": round(current_tgr, 4),
                }

            else:  # cap_rate
                if q_idx > 0:
                    lo, hi = cfg["cap_rate_noise_qoq"]
                    current_cr = max(cfg["cap_rate_min"] - 0.005,
                                     min(cfg["cap_rate_max"] + 0.005,
                                         current_cr + rng.uniform(lo, hi)))
                assump_kw = {
                    "cap_rate": round(current_cr, 4),
                }

            # Detect whether assumption changed meaningfully from prior quarter
            assumption_changed = False
            if prev_assumption is not None:
                if method == "comparable_multiples":
                    prev_m = (prev_assumption.ebitda_multiple
                              or prev_assumption.revenue_multiple or 0.0)
                    curr_m = (assump_kw.get("ebitda_multiple")
                              or assump_kw.get("revenue_multiple") or 0.0)
                    assumption_changed = abs(curr_m - prev_m) > 0.05
                elif method == "dcf":
                    assumption_changed = abs(current_dr - prev_assumption.discount_rate) > 0.001
                else:  # cap_rate
                    assumption_changed = abs(current_cr - prev_assumption.cap_rate) > 0.001

            # Build records
            assump_id = _assumption_id(holding.holding_id, quarter)
            assump = AssumptionRecord(
                assumption_id=assump_id,
                holding_id=holding.holding_id,
                fund_id=holding.fund_id,
                quarter=quarter,
                method=method,
                approval_status=approval_status,
                notes=notes,
                assumption_changed=assumption_changed,
                assumption_source=ASSUMPTION_DATA_ACCESS.get(
                    holding.fund_id, _DEFAULT_ASSUMPTION_SOURCE),
                **assump_kw,
            )
            all_assumptions.append(assump)

            fv = compute_fair_value(
                current_metric, method,
                ebitda_multiple=assump.ebitda_multiple,
                revenue_multiple=assump.revenue_multiple,
                discount_rate=assump.discount_rate,
                terminal_growth_rate=assump.terminal_growth_rate,
                cap_rate=assump.cap_rate,
            )
            val = ValuationRecord(
                valuation_id=_valuation_id(holding.holding_id, quarter),
                holding_id=holding.holding_id,
                fund_id=holding.fund_id,
                quarter=quarter,
                assumption_id=assump_id,
                underlying_metric_mm=round(current_metric, 3),
                underlying_metric_type=metric_type,
                fair_value_mm=round(fv, 3),
                currency=holding.currency,
            )
            all_valuations.append(val)
            prev_assumption = assump

    return all_assumptions, all_valuations


# ---------------------------------------------------------------------------
# Attribution (called by script 07)
# ---------------------------------------------------------------------------

def compute_bridges(
    holdings:    List[HoldingRecord],
    assumptions: List[AssumptionRecord],
    valuations:  List[ValuationRecord],
) -> List[BridgeRecord]:
    """Compute quarter-over-quarter bridges for all holdings (Q2 2024 onwards)."""
    assump_idx = {(a.holding_id, a.quarter): a for a in assumptions}
    val_idx    = {(v.holding_id, v.quarter): v for v in valuations}

    bridges = []
    for i in range(1, len(QUARTERS)):
        prev_q = QUARTERS[i - 1]
        curr_q = QUARTERS[i]
        for holding in holdings:
            hid  = holding.holding_id
            pa   = assump_idx.get((hid, prev_q))
            ca   = assump_idx.get((hid, curr_q))
            pv   = val_idx.get((hid, prev_q))
            cv   = val_idx.get((hid, curr_q))
            if pa and ca and pv and cv:
                bridges.append(compute_bridge(pv, cv, pa, ca, holding, prev_quarter=prev_q))
    return bridges


def compute_drift(
    assumptions: List[AssumptionRecord],
    valuations:  List[ValuationRecord],
    bridges:     List[BridgeRecord],
) -> List[DriftRecord]:
    """Compute portfolio-level assumption trends and bridge decomposition per quarter."""
    assump_by_q: Dict[str, List[AssumptionRecord]] = defaultdict(list)
    for a in assumptions:
        assump_by_q[a.quarter].append(a)

    val_by_q: Dict[str, List[ValuationRecord]] = defaultdict(list)
    for v in valuations:
        val_by_q[v.quarter].append(v)

    bridge_by_q: Dict[str, List[BridgeRecord]] = defaultdict(list)
    for b in bridges:
        bridge_by_q[b.quarter].append(b)

    records = []
    for quarter in QUARTERS:
        qa = assump_by_q[quarter]
        qv = val_by_q[quarter]
        qb = bridge_by_q[quarter]

        buyout_ids  = {f.fund_id for f in FUND_DEFINITIONS if f.strategy == "Buyout"}
        geq_ids     = {f.fund_id for f in FUND_DEFINITIONS if f.strategy == "Growth Equity"}
        infra_ids   = {f.fund_id for f in FUND_DEFINITIONS if f.strategy == "Infrastructure"}
        re_ids      = {f.fund_id for f in FUND_DEFINITIONS if f.strategy == "Real Estate"}

        buyout_mult = [a.ebitda_multiple  for a in qa if a.fund_id in buyout_ids and a.ebitda_multiple  is not None]
        geq_mult    = [a.revenue_multiple for a in qa if a.fund_id in geq_ids    and a.revenue_multiple is not None]
        infra_dr    = [a.discount_rate    for a in qa if a.fund_id in infra_ids  and a.discount_rate    is not None]
        re_cr       = [a.cap_rate         for a in qa if a.fund_id in re_ids     and a.cap_rate         is not None]

        def _avg(xs: list) -> float:
            return round(sum(xs) / len(xs), 4) if xs else 0.0

        def _median(xs: list) -> float:
            if not xs:
                return 0.0
            s = sorted(xs)
            n = len(s)
            if n % 2 == 1:
                return round(s[n // 2], 4)
            return round((s[n // 2 - 1] + s[n // 2]) / 2.0, 4)

        # NAV breakdown by assumption data source
        source_by_hid = {a.holding_id: a.assumption_source for a in qa}
        nav_by_source: Dict[str, float] = defaultdict(float)
        for v in qv:
            src = source_by_hid.get(v.holding_id, _DEFAULT_ASSUMPTION_SOURCE)
            nav_by_source[src] += v.fair_value_mm

        total_nav = round(sum(v.fair_value_mm for v in qv), 2)
        nav_disc  = round(nav_by_source.get("disclosed", 0.0), 2)
        nav_back  = round(nav_by_source.get("back_calculated", 0.0), 2)
        nav_extr  = round(total_nav - nav_disc - nav_back, 2)  # derive to guarantee sum = total

        records.append(DriftRecord(
            quarter=quarter,
            avg_buyout_ebitda_multiple=_avg(buyout_mult),
            median_buyout_ebitda_multiple=_median(buyout_mult),
            avg_geq_revenue_multiple=_avg(geq_mult),
            median_geq_revenue_multiple=_median(geq_mult),
            avg_infra_discount_rate=_avg(infra_dr),
            median_infra_discount_rate=_median(infra_dr),
            avg_re_cap_rate=_avg(re_cr),
            median_re_cap_rate=_median(re_cr),
            total_portfolio_nav_mm=total_nav,
            operational_change_mm=round(sum(b.operational_component_mm for b in qb), 2),
            assumption_change_mm=round(sum(b.assumption_component_mm for b in qb), 2),
            methodology_change_mm=round(sum(b.methodology_component_mm for b in qb), 2),
            nav_disclosed_mm=nav_disc,
            nav_back_calculated_mm=nav_back,
            nav_extracted_mm=nav_extr,
        ))
    return records
