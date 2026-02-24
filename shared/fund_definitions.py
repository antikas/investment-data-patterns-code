"""
SSOT for all fund identity data, terminology sets, and portfolio company definitions.

27 funds across 7 strategies, 3 quality tiers, 3 currencies, 4 terminology sets.
~100 portfolio company positions total.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortfolioCompanyDef:
    name: str
    sector: str
    investment_date: str          # YYYY-MM-DD
    initial_cost_mm: float
    ownership_pct: float


@dataclass
class FundDefinition:
    fund_id: str
    fund_name: str
    gp_name: str
    vintage_year: int
    strategy: str
    committed_capital_mm: float
    currency: str
    fund_size_mm: float           # total fund size (all LPs)
    lp_commitment_mm: float       # this LP's commitment
    report_quality_tier: str      # "institutional", "narrative", "poor"
    reporting_speed_days: int     # days after quarter end to produce report
    terminology: str              # "A", "B", "C", "D" — maps to TERMINOLOGY_SETS
    portfolio_companies: List[PortfolioCompanyDef] = field(default_factory=list)
    include_portfolio_schedule: bool = True  # institutional tier: show schedule of investments?


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUARTERS = [
    "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024",
    "Q1 2025", "Q2 2025", "Q3 2025",
]

QUARTER_END_DATES = {
    "Q1 2024": "2024-03-31",
    "Q2 2024": "2024-06-30",
    "Q3 2024": "2024-09-30",
    "Q4 2024": "2024-12-31",
    "Q1 2025": "2025-03-31",
    "Q2 2025": "2025-06-30",
    "Q3 2025": "2025-09-30",
}

FX_RATES = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27}
FX_RATE_DATE = "2025-09-30"


# ---------------------------------------------------------------------------
# Terminology sets (4 sets, assigned per GP)
# ---------------------------------------------------------------------------

TERMINOLOGY_SETS = {
    "A": {  # ILPA standard
        "nav": "Net Asset Value",
        "committed": "Total Commitments",
        "called": "Capital Called",
        "distributed": "Distributions",
        "irr_net": "Net IRR",
        "irr_gross": "Gross IRR",
        "tvpi": "TVPI",
        "dpi": "DPI",
        "rvpi": "RVPI",
        "management_fee": "Management Fee",
        "carried_interest": "Carried Interest",
        "fund_summary_title": "Fund Summary",
        "performance_title": "Performance Metrics (Net of Fees)",
        "schedule_title": "Schedule of Investments",
        "fees_title": "Fee & Expense Summary",
        "capital_account_title": "Partner's Capital Account Statement",
    },
    "B": {  # Traditional
        "nav": "LP Interest Value",
        "committed": "Fund Size",
        "called": "Drawn Capital",
        "distributed": "Capital Returned",
        "irr_net": "Net Internal Rate of Return",
        "irr_gross": "Gross Internal Rate of Return",
        "tvpi": "Total Value to Paid-In",
        "dpi": "Distributions to Paid-In",
        "rvpi": "Residual Value to Paid-In",
        "management_fee": "Management Fee",
        "carried_interest": "Performance Allocation",
        "fund_summary_title": "Fund Overview",
        "performance_title": "Investment Performance",
        "schedule_title": "Portfolio Holdings",
        "fees_title": "Fee Summary",
        "capital_account_title": "Capital Account Summary",
    },
    "C": {  # Formal / Legal
        "nav": "Partner's Capital Account Balance",
        "committed": "Aggregate Commitments",
        "called": "Capital Contributions",
        "distributed": "Cumulative Distributions",
        "irr_net": "Net IRR Since Inception",
        "irr_gross": "Gross IRR Since Inception",
        "tvpi": "Total Value Multiple",
        "dpi": "Realisation Multiple",
        "rvpi": "Unrealised Multiple",
        "management_fee": "Management Fee",
        "carried_interest": "Carried Interest Allocation",
        "fund_summary_title": "Summary of Fund Activity",
        "performance_title": "Performance Summary",
        "schedule_title": "Schedule of Portfolio Investments",
        "fees_title": "Fees & Expenses",
        "capital_account_title": "Statement of Partner's Capital",
    },
    "D": {  # Casual / Informal
        "nav": "NAV",
        "committed": "Committed Capital",
        "called": "Called",
        "distributed": "Returned",
        "irr_net": "IRR",
        "irr_gross": "Gross IRR",
        "tvpi": "Multiple",
        "dpi": "DPI",
        "rvpi": "RVPI",
        "management_fee": "Mgmt Fee",
        "carried_interest": "Carry",
        "fund_summary_title": "Fund Summary",
        "performance_title": "Returns",
        "schedule_title": "Investments",
        "fees_title": "Fees",
        "capital_account_title": "Capital Account",
    },
}

# Currency symbols for PDF formatting
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3"}


# ---------------------------------------------------------------------------
# GP addresses (one per GP, for letterhead variation)
# ---------------------------------------------------------------------------

GP_ADDRESSES = {
    "Apex Capital Management": "200 Clarendon Street, 52nd Floor | Boston, MA 02116 | Tel: (617) 555-0190",
    "Vanguard Infrastructure Partners": "Beethovenstrasse 24 | 8002 Zurich, Switzerland | Tel: +41 44 555 0130",
    "Summit Capital Partners": "One Vanderbilt Avenue, 40th Floor | New York, NY 10017 | Tel: (212) 555-0210",
    "Elevate Capital Group": "1 Market Street, Suite 3600 | San Francisco, CA 94105 | Tel: (415) 555-0175",
    "Rheinland Asset Management": "Koenigsallee 60 | 40212 Duesseldorf, Germany | Tel: +49 211 555 0145",
    "Catalyst Secondaries Partners": "600 Fifth Avenue, 30th Floor | New York, NY 10020 | Tel: (212) 555-0188",
    "Redwood Ventures": "2500 Sand Hill Road, Suite 200 | Menlo Park, CA 94025 | Tel: (650) 555-0162",
    "Albion Credit Partners": "25 Old Broad Street, 12th Floor | London EC2N 1HQ | Tel: +44 20 7555 0155",
    "Oakmont Private Equity": "320 Park Avenue, 18th Floor | New York, NY 10022 | Tel: (212) 555-0134",
    "Atlas Real Assets Management": "100 Federal Street, 28th Floor | Boston, MA 02110 | Tel: (617) 555-0177",
    "Schwarzwald Capital": "Maximilianstrasse 35 | 80539 Munich, Germany | Tel: +49 89 555 0168",
    "Venture Works Capital": "600 Congress Avenue, Suite 1400 | Austin, TX 78701 | Tel: (512) 555-0143",
    "Cornerstone Property Advisors": "350 Madison Avenue, 22nd Floor | New York, NY 10017 | Tel: (212) 555-0156",
    "Horizon Growth Partners": "1999 Avenue of the Stars, Suite 2700 | Los Angeles, CA 90067 | Tel: (310) 555-0182",
    "Britannia Capital Partners": "1 Finsbury Avenue, 15th Floor | London EC2M 2PF | Tel: +44 20 7555 0191",
    "Pinnacle Real Estate Advisors": "227 West Monroe Street, Suite 4000 | Chicago, IL 60606 | Tel: (312) 555-0147",
    "Emerging Ventures Management": "55 East 52nd Street, 35th Floor | New York, NY 10055 | Tel: (212) 555-0163",
    "Northstar Buyout Partners": "30 Rockefeller Plaza, 26th Floor | New York, NY 10112 | Tel: (212) 555-0199",
    "Pacific Credit Advisors": "555 California Street, Suite 3200 | San Francisco, CA 94104 | Tel: (415) 555-0188",
    "Continental Growth Partners": "Avenue Louise 480 | 1050 Brussels, Belgium | Tel: +32 2 555 0176",
    "Micro Capital Partners": "7 World Trade Center, 44th Floor | New York, NY 10007 | Tel: (212) 555-0112",
    "Landmark Secondaries Group": "399 Park Avenue, 16th Floor | New York, NY 10022 | Tel: (212) 555-0205",
    "Westcoast Ventures": "1000 Wilshire Boulevard, Suite 2100 | Los Angeles, CA 90017 | Tel: (213) 555-0171",
    "Meridian Infrastructure Partners": "125 High Street, 21st Floor | Boston, MA 02110 | Tel: (617) 555-0196",
    "Titan Capital Partners": "383 Madison Avenue, 24th Floor | New York, NY 10179 | Tel: (212) 555-0228",
    "Seed & Scale Ventures": "144 Townsend Street, Suite 200 | San Francisco, CA 94107 | Tel: (415) 555-0153",
    "Compass Credit Management": "Two International Place, 19th Floor | Boston, MA 02110 | Tel: (617) 555-0214",
}


# ---------------------------------------------------------------------------
# 27 Fund Definitions
# ---------------------------------------------------------------------------

FUND_DEFINITIONS: List[FundDefinition] = [
    # -----------------------------------------------------------------------
    # 1. BYT-2017-I — Mature buyout, harvest period
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2017-I",
        fund_name="Apex Buyout Partners VII",
        gp_name="Apex Capital Management",
        vintage_year=2017,
        strategy="Buyout",
        committed_capital_mm=150.0,
        currency="USD",
        fund_size_mm=150.0,
        lp_commitment_mm=7.5,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Meridian Healthcare Group", "Healthcare Services", "2017-06-15", 185.0, 8.2),
            PortfolioCompanyDef("Transcend Data Solutions", "Enterprise Software", "2017-11-20", 142.0, 6.3),
            PortfolioCompanyDef("Ironclad Security Systems", "Cybersecurity", "2018-03-10", 128.0, 5.7),
            PortfolioCompanyDef("Pinnacle Consumer Brands", "Consumer Products", "2018-09-05", 165.0, 7.3),
        ],
    ),
    # -----------------------------------------------------------------------
    # 2. INF-2017-I — Mature infrastructure, EUR, steady distributions
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="INF-2017-I",
        fund_name="Vanguard European Infrastructure III",
        gp_name="Vanguard Infrastructure Partners",
        vintage_year=2017,
        strategy="Infrastructure",
        committed_capital_mm=80.0,
        currency="EUR",
        fund_size_mm=80.0,
        lp_commitment_mm=5.0,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="B",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Clearwater Utilities", "Water & Waste", "2017-09-01", 120.0, 10.0),
            PortfolioCompanyDef("Nordic Wind Holdings", "Renewables", "2018-02-15", 95.0, 7.9),
            PortfolioCompanyDef("TransAlpine Toll Roads", "Transportation", "2018-07-20", 88.0, 7.3),
            PortfolioCompanyDef("Rhine Digital Networks", "Telecommunications", "2019-01-10", 72.0, 6.0),
        ],
    ),
    # -----------------------------------------------------------------------
    # 3. BYT-2018-I — Large flagship buyout, approaching harvest
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2018-I",
        fund_name="Summit Capital Partners Fund V",
        gp_name="Summit Capital Partners",
        vintage_year=2018,
        strategy="Buyout",
        committed_capital_mm=200.0,
        currency="USD",
        fund_size_mm=200.0,
        lp_commitment_mm=10.0,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Atlas Manufacturing Corp", "Industrials", "2018-04-20", 210.0, 7.0),
            PortfolioCompanyDef("Coastal Hospitality Group", "Hospitality", "2018-10-15", 175.0, 5.8),
            PortfolioCompanyDef("Vertex Automotive", "Automotive Parts", "2019-03-01", 195.0, 6.5),
            PortfolioCompanyDef("Keystone Financial Services", "Financial Services", "2019-08-12", 160.0, 5.3),
            PortfolioCompanyDef("Ridgeline Pharma", "Pharmaceuticals", "2020-01-25", 140.0, 4.7),
        ],
    ),
    # -----------------------------------------------------------------------
    # 4. GEQ-2018-I — Growth equity, narrative tier, formal terms
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="GEQ-2018-I",
        fund_name="Elevate Growth Equity Fund III",
        gp_name="Elevate Capital Group",
        vintage_year=2018,
        strategy="Growth Equity",
        committed_capital_mm=60.0,
        currency="USD",
        fund_size_mm=60.0,
        lp_commitment_mm=3.5,
        report_quality_tier="narrative",
        reporting_speed_days=45,
        terminology="C",
        portfolio_companies=[
            PortfolioCompanyDef("CloudSync Platform", "Cloud Infrastructure", "2018-07-10", 65.0, 12.8),
            PortfolioCompanyDef("FinEdge Technologies", "Fintech", "2019-01-20", 55.0, 10.8),
            PortfolioCompanyDef("MedTech Innovations", "Healthcare IT", "2019-06-15", 48.0, 9.4),
            PortfolioCompanyDef("DataForge Analytics", "Data & AI", "2020-02-10", 42.0, 8.2),
        ],
    ),
    # -----------------------------------------------------------------------
    # 5. REA-2018-I — Real estate, EUR, narrative tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="REA-2018-I",
        fund_name="Rheinland European Real Estate IV",
        gp_name="Rheinland Asset Management",
        vintage_year=2018,
        strategy="Real Estate",
        committed_capital_mm=45.0,
        currency="EUR",
        fund_size_mm=45.0,
        lp_commitment_mm=3.0,
        report_quality_tier="narrative",
        reporting_speed_days=60,
        terminology="B",
        portfolio_companies=[
            PortfolioCompanyDef("Berlin Office Portfolio", "Office", "2018-08-01", 68.0, 14.2),
            PortfolioCompanyDef("Munich Logistics Centre", "Industrial", "2019-02-15", 55.0, 11.5),
            PortfolioCompanyDef("Amsterdam Mixed-Use Development", "Mixed Use", "2019-09-10", 48.0, 10.0),
            PortfolioCompanyDef("Paris Retail Complex", "Retail", "2020-04-20", 38.0, 7.9),
        ],
    ),
    # -----------------------------------------------------------------------
    # 6. SEC-2019-I — Secondaries, institutional, fast deployment
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="SEC-2019-I",
        fund_name="Catalyst Secondaries Opportunities IV",
        gp_name="Catalyst Secondaries Partners",
        vintage_year=2019,
        strategy="Secondaries",
        committed_capital_mm=90.0,
        currency="USD",
        fund_size_mm=90.0,
        lp_commitment_mm=5.0,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Acquired LP Portfolio I", "Diversified Buyout", "2019-06-01", 145.0, 16.1),
            PortfolioCompanyDef("Acquired LP Portfolio II", "Growth / Venture", "2019-10-15", 120.0, 13.3),
            PortfolioCompanyDef("GP-Led Continuation Fund Alpha", "Buyout", "2020-03-20", 95.0, 10.6),
            PortfolioCompanyDef("Acquired LP Portfolio III", "Infrastructure", "2020-08-10", 85.0, 9.4),
        ],
    ),
    # -----------------------------------------------------------------------
    # 7. VCA-2019-I — Venture capital, narrative tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="VCA-2019-I",
        fund_name="Redwood Ventures Fund III",
        gp_name="Redwood Ventures",
        vintage_year=2019,
        strategy="Venture Capital",
        committed_capital_mm=35.0,
        currency="USD",
        fund_size_mm=35.0,
        lp_commitment_mm=2.0,
        report_quality_tier="narrative",
        reporting_speed_days=45,
        terminology="C",
        portfolio_companies=[
            PortfolioCompanyDef("NovaBio Therapeutics", "Biotech", "2019-08-01", 38.0, 15.2),
            PortfolioCompanyDef("QuantumLeap Computing", "Deep Tech", "2020-01-15", 32.0, 12.8),
            PortfolioCompanyDef("GreenShift Energy", "Climate Tech", "2020-06-20", 28.0, 11.2),
            PortfolioCompanyDef("Lumen AI", "Enterprise AI", "2021-01-10", 22.0, 8.8),
            PortfolioCompanyDef("FairTrade Digital", "Fintech", "2021-07-01", 18.0, 7.2),
        ],
    ),
    # -----------------------------------------------------------------------
    # 8. PCR-2019-I — Private credit, GBP, narrative tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="PCR-2019-I",
        fund_name="Albion Direct Lending Fund II",
        gp_name="Albion Credit Partners",
        vintage_year=2019,
        strategy="Private Credit",
        committed_capital_mm=50.0,
        currency="GBP",
        fund_size_mm=50.0,
        lp_commitment_mm=3.0,
        report_quality_tier="narrative",
        reporting_speed_days=60,
        terminology="B",
        portfolio_companies=[
            PortfolioCompanyDef("Sterling Manufacturing", "Industrials", "2019-09-15", 52.0, 10.4),
            PortfolioCompanyDef("Britannia Healthcare", "Healthcare", "2020-02-01", 45.0, 9.0),
            PortfolioCompanyDef("Trent Valley Logistics", "Logistics", "2020-07-20", 40.0, 8.0),
            PortfolioCompanyDef("Highland Retail Finance", "Financial Services", "2021-01-10", 35.0, 7.0),
        ],
    ),
    # -----------------------------------------------------------------------
    # 9. BYT-2019-I — Mid-size buyout, narrative, casual terms
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2019-I",
        fund_name="Oakmont Buyout Opportunities III",
        gp_name="Oakmont Private Equity",
        vintage_year=2019,
        strategy="Buyout",
        committed_capital_mm=100.0,
        currency="USD",
        fund_size_mm=100.0,
        lp_commitment_mm=5.0,
        report_quality_tier="narrative",
        reporting_speed_days=45,
        terminology="D",
        portfolio_companies=[
            PortfolioCompanyDef("Horizon Media Group", "Media & Entertainment", "2019-05-20", 95.0, 9.5),
            PortfolioCompanyDef("Cascade Food Services", "Food & Beverage", "2019-11-10", 82.0, 8.2),
            PortfolioCompanyDef("Pinnacle Logistics", "Supply Chain", "2020-04-15", 75.0, 7.5),
            PortfolioCompanyDef("Summit Technology Partners", "Technology Services", "2020-10-01", 68.0, 6.8),
        ],
    ),
    # -----------------------------------------------------------------------
    # 10. INF-2020-I — Infrastructure, institutional, J-curve inflection
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="INF-2020-I",
        fund_name="Atlas Infrastructure Fund IV",
        gp_name="Atlas Real Assets Management",
        vintage_year=2020,
        strategy="Infrastructure",
        committed_capital_mm=120.0,
        currency="USD",
        fund_size_mm=120.0,
        lp_commitment_mm=7.5,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("ClearWater Utility Holdings", "Water & Waste", "2020-07-15", 155.0, 12.9),
            PortfolioCompanyDef("Portside Terminal Group", "Transportation", "2020-12-01", 130.0, 10.8),
            PortfolioCompanyDef("GridBridge Energy", "Power Distribution", "2021-05-20", 115.0, 9.6),
            PortfolioCompanyDef("SolarPeak Generation", "Renewables", "2021-11-10", 100.0, 8.3),
        ],
    ),
    # -----------------------------------------------------------------------
    # 11. BYT-2020-I — EUR buyout, institutional
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2020-I",
        fund_name="Schwarzwald European Buyout IV",
        gp_name="Schwarzwald Capital",
        vintage_year=2020,
        strategy="Buyout",
        committed_capital_mm=180.0,
        currency="EUR",
        fund_size_mm=180.0,
        lp_commitment_mm=9.0,
        report_quality_tier="institutional",
        reporting_speed_days=45,
        terminology="B",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Eurotech Industrial AG", "Industrials", "2020-09-01", 195.0, 7.2),
            PortfolioCompanyDef("HealthFirst Europe GmbH", "Healthcare", "2021-02-15", 170.0, 6.3),
            PortfolioCompanyDef("TechVision Software AG", "Enterprise Software", "2021-07-20", 155.0, 5.7),
            PortfolioCompanyDef("Alpine Consumer Group", "Consumer Products", "2022-01-10", 135.0, 5.0),
        ],
    ),
    # -----------------------------------------------------------------------
    # 12. VCA-2020-I — Venture, poor tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="VCA-2020-I",
        fund_name="Venture Works Fund II",
        gp_name="Venture Works Capital",
        vintage_year=2020,
        strategy="Venture Capital",
        committed_capital_mm=30.0,
        currency="USD",
        fund_size_mm=30.0,
        lp_commitment_mm=1.5,
        report_quality_tier="poor",
        reporting_speed_days=75,
        terminology="D",
        portfolio_companies=[
            PortfolioCompanyDef("NovaBio Inc.", "Biotech", "2020-08-15", 28.0, 14.0),
            PortfolioCompanyDef("CloudNine Robotics", "Robotics", "2021-01-20", 25.0, 12.5),
            PortfolioCompanyDef("ZeroCarbon Materials", "Climate Tech", "2021-06-10", 22.0, 11.0),
            PortfolioCompanyDef("SwiftPay Solutions", "Fintech", "2022-01-05", 18.0, 9.0),
        ],
    ),
    # -----------------------------------------------------------------------
    # 13. REA-2020-I — Real estate, poor tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="REA-2020-I",
        fund_name="Cornerstone Value-Add Real Estate II",
        gp_name="Cornerstone Property Advisors",
        vintage_year=2020,
        strategy="Real Estate",
        committed_capital_mm=35.0,
        currency="USD",
        fund_size_mm=35.0,
        lp_commitment_mm=2.0,
        report_quality_tier="poor",
        reporting_speed_days=90,
        terminology="D",
        portfolio_companies=[
            PortfolioCompanyDef("Sunbelt Office Park", "Office", "2020-11-01", 45.0, 17.1),
            PortfolioCompanyDef("Metro Living Apartments", "Multifamily", "2021-04-15", 38.0, 14.5),
            PortfolioCompanyDef("Lakeside Retail Center", "Retail", "2021-10-20", 32.0, 12.2),
        ],
    ),
    # -----------------------------------------------------------------------
    # 14. GEQ-2021-I — Growth equity, institutional
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="GEQ-2021-I",
        fund_name="Horizon Growth Partners IV",
        gp_name="Horizon Growth Partners",
        vintage_year=2021,
        strategy="Growth Equity",
        committed_capital_mm=85.0,
        currency="USD",
        fund_size_mm=85.0,
        lp_commitment_mm=5.0,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Helios Data Systems", "Enterprise Software", "2021-09-15", 85.0, 12.4),
            PortfolioCompanyDef("Canopy Health", "Healthcare IT", "2022-01-20", 62.0, 8.7),
            PortfolioCompanyDef("Ridgeline Logistics Tech", "Supply Chain & Logistics", "2022-06-10", 71.0, 9.8),
            PortfolioCompanyDef("Prism Analytics", "Data & AI", "2023-03-01", 55.0, 7.2),
            PortfolioCompanyDef("Vertex Manufacturing", "Industrial Technology", "2023-08-22", 48.0, 6.5),
        ],
    ),
    # -----------------------------------------------------------------------
    # 15. BYT-2021-I — GBP buyout, institutional, formal terms
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2021-I",
        fund_name="Britannia Buyout Fund VI",
        gp_name="Britannia Capital Partners",
        vintage_year=2021,
        strategy="Buyout",
        committed_capital_mm=110.0,
        currency="GBP",
        fund_size_mm=110.0,
        lp_commitment_mm=5.5,
        report_quality_tier="institutional",
        reporting_speed_days=45,
        terminology="C",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Thames Engineering Group", "Industrials", "2021-06-01", 115.0, 8.4),
            PortfolioCompanyDef("Canterbury Healthcare", "Healthcare Services", "2021-11-15", 98.0, 7.1),
            PortfolioCompanyDef("York Digital Media", "Media & Entertainment", "2022-04-20", 85.0, 6.2),
            PortfolioCompanyDef("Bristol Consumer Holdings", "Consumer Products", "2022-10-01", 72.0, 5.2),
        ],
    ),
    # -----------------------------------------------------------------------
    # 16. REA-2021-I — Real estate, narrative
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="REA-2021-I",
        fund_name="Pinnacle Core-Plus Real Estate V",
        gp_name="Pinnacle Real Estate Advisors",
        vintage_year=2021,
        strategy="Real Estate",
        committed_capital_mm=60.0,
        currency="USD",
        fund_size_mm=60.0,
        lp_commitment_mm=3.5,
        report_quality_tier="narrative",
        reporting_speed_days=60,
        terminology="B",
        portfolio_companies=[
            PortfolioCompanyDef("Harborview Office Complex", "Office", "2021-08-15", 78.0, 13.0),
            PortfolioCompanyDef("Sunbelt Industrial Park", "Industrial", "2022-01-10", 65.0, 10.8),
            PortfolioCompanyDef("Metro Living Residences", "Multifamily", "2022-07-20", 58.0, 9.7),
            PortfolioCompanyDef("Pacific Retail Portfolio", "Retail", "2023-02-05", 45.0, 7.5),
        ],
    ),
    # -----------------------------------------------------------------------
    # 17. GEQ-2021-II — Small growth equity, poor tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="GEQ-2021-II",
        fund_name="Emerging Growth Opportunities I",
        gp_name="Emerging Ventures Management",
        vintage_year=2021,
        strategy="Growth Equity",
        committed_capital_mm=40.0,
        currency="USD",
        fund_size_mm=40.0,
        lp_commitment_mm=2.0,
        report_quality_tier="poor",
        reporting_speed_days=90,
        terminology="D",
        portfolio_companies=[
            PortfolioCompanyDef("TechFlow Solutions", "Software", "2021-10-01", 42.0, 13.1),
            PortfolioCompanyDef("GreenLeaf Organics", "Consumer", "2022-03-15", 35.0, 10.9),
            PortfolioCompanyDef("DataVault Inc", "Data Services", "2022-08-01", 30.0, 9.4),
        ],
    ),
    # -----------------------------------------------------------------------
    # 18. BYT-2022-I — Large 2022 vintage buyout, J-curve
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2022-I",
        fund_name="Northstar Buyout Partners VIII",
        gp_name="Northstar Buyout Partners",
        vintage_year=2022,
        strategy="Buyout",
        committed_capital_mm=200.0,
        currency="USD",
        fund_size_mm=200.0,
        lp_commitment_mm=10.0,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Helios Data", "Enterprise Software", "2022-06-15", 175.0, 7.3),
            PortfolioCompanyDef("Osprey Media Group", "Media & Entertainment", "2022-11-01", 155.0, 6.5),
            PortfolioCompanyDef("Thornton Healthcare Services", "Healthcare Services", "2023-04-20", 140.0, 5.8),
            PortfolioCompanyDef("Cascade Consumer Brands", "Consumer Products", "2023-09-10", 120.0, 5.0),
        ],
    ),
    # -----------------------------------------------------------------------
    # 19. PCR-2022-I — Private credit, narrative, formal terms
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="PCR-2022-I",
        fund_name="Pacific Direct Lending III",
        gp_name="Pacific Credit Advisors",
        vintage_year=2022,
        strategy="Private Credit",
        committed_capital_mm=50.0,
        currency="USD",
        fund_size_mm=50.0,
        lp_commitment_mm=3.0,
        report_quality_tier="narrative",
        reporting_speed_days=45,
        terminology="C",
        portfolio_companies=[
            PortfolioCompanyDef("Heritage Food Group", "Consumer Staples", "2022-08-01", 48.0, 9.6),
            PortfolioCompanyDef("Apex Software Solutions", "Technology", "2023-01-15", 42.0, 8.4),
            PortfolioCompanyDef("Westfield Industrial", "Industrials", "2023-06-20", 38.0, 7.6),
        ],
    ),
    # -----------------------------------------------------------------------
    # 20. GEQ-2022-I — EUR growth equity, narrative
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="GEQ-2022-I",
        fund_name="Continental Growth Fund II",
        gp_name="Continental Growth Partners",
        vintage_year=2022,
        strategy="Growth Equity",
        committed_capital_mm=40.0,
        currency="EUR",
        fund_size_mm=40.0,
        lp_commitment_mm=2.5,
        report_quality_tier="narrative",
        reporting_speed_days=60,
        terminology="B",
        portfolio_companies=[
            PortfolioCompanyDef("BrightPath Education Tech", "EdTech", "2022-09-15", 45.0, 13.2),
            PortfolioCompanyDef("NordicSaaS Platform", "SaaS", "2023-02-20", 38.0, 11.2),
            PortfolioCompanyDef("MediterraneanFood Co", "Food & Beverage", "2023-08-10", 32.0, 9.4),
        ],
    ),
    # -----------------------------------------------------------------------
    # 21. BYT-2022-II — Small buyout, poor tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2022-II",
        fund_name="Micro Buyout Partners II",
        gp_name="Micro Capital Partners",
        vintage_year=2022,
        strategy="Buyout",
        committed_capital_mm=30.0,
        currency="USD",
        fund_size_mm=30.0,
        lp_commitment_mm=1.5,
        report_quality_tier="poor",
        reporting_speed_days=90,
        terminology="D",
        portfolio_companies=[
            PortfolioCompanyDef("Redwood Properties", "Real Estate Services", "2022-10-01", 35.0, 15.6),
            PortfolioCompanyDef("Coastal Shipping Co", "Logistics", "2023-03-15", 30.0, 13.3),
            PortfolioCompanyDef("Mountain View Dental Group", "Healthcare Services", "2023-09-01", 25.0, 11.1),
        ],
    ),
    # -----------------------------------------------------------------------
    # 22. SEC-2023-I — Secondaries, narrative, early
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="SEC-2023-I",
        fund_name="Landmark Secondaries Fund VI",
        gp_name="Landmark Secondaries Group",
        vintage_year=2023,
        strategy="Secondaries",
        committed_capital_mm=75.0,
        currency="USD",
        fund_size_mm=75.0,
        lp_commitment_mm=4.0,
        report_quality_tier="narrative",
        reporting_speed_days=45,
        terminology="C",
        portfolio_companies=[
            PortfolioCompanyDef("Acquired LP Portfolio Alpha", "Diversified PE", "2023-06-01", 115.0, 15.3),
            PortfolioCompanyDef("Acquired LP Portfolio Beta", "Growth / Venture", "2023-10-15", 85.0, 11.3),
            PortfolioCompanyDef("GP-Led Continuation Vehicle I", "Buyout", "2024-02-01", 65.0, 8.7),
        ],
    ),
    # -----------------------------------------------------------------------
    # 23. VCA-2023-I — Venture, poor tier, very early
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="VCA-2023-I",
        fund_name="Westcoast Ventures Fund I",
        gp_name="Westcoast Ventures",
        vintage_year=2023,
        strategy="Venture Capital",
        committed_capital_mm=25.0,
        currency="USD",
        fund_size_mm=25.0,
        lp_commitment_mm=1.2,
        report_quality_tier="poor",
        reporting_speed_days=75,
        terminology="D",
        portfolio_companies=[
            PortfolioCompanyDef("NeuroLink AI", "Enterprise AI", "2023-07-15", 22.0, 13.8),
            PortfolioCompanyDef("BioSynth Labs", "Biotech", "2023-12-01", 18.0, 11.3),
            PortfolioCompanyDef("EcoCharge Systems", "Climate Tech", "2024-03-10", 15.0, 9.4),
        ],
    ),
    # -----------------------------------------------------------------------
    # 24. INF-2023-I — Infrastructure, institutional
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="INF-2023-I",
        fund_name="Meridian Infrastructure Fund II",
        gp_name="Meridian Infrastructure Partners",
        vintage_year=2023,
        strategy="Infrastructure",
        committed_capital_mm=60.0,
        currency="USD",
        fund_size_mm=60.0,
        lp_commitment_mm=3.5,
        report_quality_tier="institutional",
        reporting_speed_days=45,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Cascadia Water Systems", "Water & Waste", "2023-08-01", 65.0, 10.8),
            PortfolioCompanyDef("Heartland Power Grid", "Power Distribution", "2024-01-15", 55.0, 9.2),
            PortfolioCompanyDef("Coastal Fibre Networks", "Telecommunications", "2024-06-01", 45.0, 7.5),
        ],
    ),
    # -----------------------------------------------------------------------
    # 25. BYT-2024-I — Very recent buyout, deep J-curve
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="BYT-2024-I",
        fund_name="Titan Capital Partners Fund IX",
        gp_name="Titan Capital Partners",
        vintage_year=2024,
        strategy="Buyout",
        committed_capital_mm=150.0,
        currency="USD",
        fund_size_mm=150.0,
        lp_commitment_mm=7.5,
        report_quality_tier="institutional",
        reporting_speed_days=30,
        terminology="A",
        include_portfolio_schedule=True,
        portfolio_companies=[
            PortfolioCompanyDef("Nexus Healthcare Systems", "Healthcare", "2024-05-15", 130.0, 5.4),
            PortfolioCompanyDef("Vantage Industrial Group", "Industrials", "2024-09-01", 110.0, 4.6),
        ],
    ),
    # -----------------------------------------------------------------------
    # 26. GEQ-2024-I — Very early growth equity, poor tier
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="GEQ-2024-I",
        fund_name="Seed & Scale Growth Fund I",
        gp_name="Seed & Scale Ventures",
        vintage_year=2024,
        strategy="Growth Equity",
        committed_capital_mm=20.0,
        currency="USD",
        fund_size_mm=20.0,
        lp_commitment_mm=1.0,
        report_quality_tier="poor",
        reporting_speed_days=90,
        terminology="D",
        portfolio_companies=[
            PortfolioCompanyDef("QuickCommerce Inc", "E-Commerce", "2024-06-01", 18.0, 12.0),
            PortfolioCompanyDef("UrbanFarm Technologies", "AgTech", "2024-09-15", 15.0, 10.0),
        ],
    ),
    # -----------------------------------------------------------------------
    # 27. PCR-2024-I — Recent private credit, narrative
    # -----------------------------------------------------------------------
    FundDefinition(
        fund_id="PCR-2024-I",
        fund_name="Compass Direct Lending Fund I",
        gp_name="Compass Credit Management",
        vintage_year=2024,
        strategy="Private Credit",
        committed_capital_mm=35.0,
        currency="USD",
        fund_size_mm=35.0,
        lp_commitment_mm=2.0,
        report_quality_tier="narrative",
        reporting_speed_days=60,
        terminology="B",
        portfolio_companies=[
            PortfolioCompanyDef("Precision Manufacturing Co", "Industrials", "2024-04-15", 35.0, 10.0),
            PortfolioCompanyDef("National Logistics Services", "Logistics", "2024-08-01", 28.0, 8.0),
            PortfolioCompanyDef("Summit Dental Holdings", "Healthcare Services", "2024-11-10", 22.0, 6.3),
        ],
    ),
]
