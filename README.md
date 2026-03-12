# Investment Data Patterns

Companion code for the article series on recurring data patterns in alternative investments.

This repo demonstrates two patterns so far:

- **Pattern 1: Structured-from-Unstructured.** LLM-powered document extraction with field-level confidence scoring, isotonic calibration, routing thresholds, and full source lineage. Processes synthetic GP quarterly reports, capital call notices, and distribution notices across three quality tiers.
- **Pattern 2: Mark-to-Model Governance.** When there's no market price, valuations are model outputs. Bridge decomposition separates NAV changes into operational, assumption, and methodology components. Surfaces assumption drift, hidden outperformance, and cross-GP divergence that independent valuation alone cannot catch.

## What's here

### Databricks notebooks

| # | Notebook | Pattern | What it does |
|---|----------|---------|--------------|
| 00 | `00_cleanup.py` | Both | Drops all tables and files. Idempotent, safe to run at any point. |
| 01 | `01_generate_synthetic_reports.py` | 1 | Generates ~450 synthetic PDF documents (quarterly reports, capital calls, distributions) across 27 funds, 7 quarters, 3 quality tiers. Writes ground truth to Delta tables. |
| 02 | `02_extract_with_confidence.py` | 1 | PDF to LLM vision extraction with field-level confidence and source lineage. Supports parallel extraction and resume mode. |
| 03 | `03_calibration_and_routing.py` | 1 | Isotonic regression calibration, routing threshold trade-off analysis, review queue simulation. |
| 04 | `04_cost_model.py` | 1 | Rules-based vs LLM extraction economics, parameterised crossover model, 3-year TCO. |
| 05 | `05_portfolio_dashboard.py` | 1 | Dataset exploration queries for the Lakeview portfolio dashboard. |
| 06 | `06_pipeline_metrics.lvdash.json` | 1 | Lakeview dashboard: extraction accuracy, calibration curves, routing decisions, review queue, source lineage, cost model. |
| 07 | `07_generate_valuation_history.py` | 2 | Synthetic valuation data: 69 holdings across 4 strategies, 7 quarters. Four governance scenarios (methodology swing, hidden outperformance, systematic drift, cross-GP divergence). |
| 08 | `08_valuation_attribution.py` | 2 | Bridge decomposition (operational vs assumption vs methodology) and portfolio-level drift analysis. |
| 09 | `09_mark_to_model_governance.lvdash.json` | 2 | Lakeview dashboard: assumption drift, attribution bridges, cross-GP divergence, flagged assumptions. |
| -- | `fast_rebuild.py` | Both | Drops and rebuilds all tables from both patterns without regenerating PDFs or calling an LLM. |

### Local scripts (development, no cluster needed)

| File | Pattern | What it does |
|------|---------|--------------|
| `01_generate_synthetic_reports.py` | 1 | Same generation logic, outputs PDFs + JSON locally |
| `02_extract_with_confidence.py` | 1 | LLM API extraction with `--concurrency N`, `--resume`, `--sample N` |
| `03_calibration_and_routing.py` | 1 | Per-document-type isotonic calibration and routing threshold sweep |
| `04_cost_model.py` | 1 | Rules-based vs LLM cost model with multi-document economics |
| `05_portfolio_dashboard.py` | 1 | Self-contained HTML dashboard with embedded charts |
| `07_generate_valuation_history.py` | 2 | Same generation as notebook 07, outputs JSON locally |
| `08_valuation_attribution.py` | 2 | Bridge computation + self-contained HTML governance dashboard |
| `run_pipeline.py` | Both | Runs downstream steps. `--pattern 1` (default): 03-05. `--pattern 2`: 07-08. `--pattern all`: both. |

### Shared modules

All domain logic lives in `shared/`, imported by both local scripts and Databricks notebooks.

| Module | Pattern | Purpose |
|--------|---------|---------|
| `fund_definitions.py` | 1 | 27-fund universe: strategies, vintages, committed capital, quality tiers |
| `simulation.py` | 1 | Deterministic fund performance simulation (J-curve, distributions, IRR) |
| `report_generators.py` | 1 | PDF generation with reportlab (3 quality tiers x 3 document types) |
| `notice_generators.py` | 1 | Capital call and distribution notice PDF generation |
| `failure_modes.py` | 1 | The 7 GP report failure modes: format, timing, terminology, currency, restatements, transcription errors, number ambiguity |
| `schemas.py` | 1 | Extraction schemas per document type (SSOT for prompts and accuracy measurement) |
| `extraction_utils.py` | 1 | Flat row to nested extraction dict reconstruction |
| `valuation_models.py` | 2 | Mark-to-model valuation: holdings generation, scenario configs, bridge decomposition, drift computation |

## Synthetic datasets

### Pattern 1: Extraction pipeline

**27 funds** across 7 strategies (buyout, growth equity, infrastructure, real estate, private credit, secondaries, venture capital). **7 quarters** (Q1 2024 - Q3 2025). **3 quality tiers** reflecting the real-world reporting spectrum:

- **Institutional**: Clean tabular layout, consistent fields (large GP)
- **Narrative**: Numbers embedded in quarterly letter prose (mid-market GP)
- **Poor quality**: Sparse, inconsistent structure (small manager)

Three document types per fund per quarter: quarterly reports, capital call notices, and distribution notices. The dataset includes the 7 common GP report failure modes: format variation, timing variation, terminology variation, currency variation, silent restatements, transcription errors, and number ambiguity.

### Pattern 2: Mark-to-model governance

**69 holdings** across 4 strategies (buyout, growth equity, infrastructure, real estate). **7 quarters** (Q1 2024 - Q3 2025). Four governance scenarios embedded in the synthetic data:

1. **Methodology-driven swing**: post-ECB rate cut repricing drives large assumption changes in a European buyout fund
2. **Hidden outperformance**: strong revenue growth masked by simultaneous multiple tightening, producing a flat net NAV change
3. **Systematic drift**: gradual EBITDA multiple expansion across all buyout GPs, tracking public market sentiment
4. **Cross-GP divergence**: two GPs valuing comparable media-sector holdings at materially different multiples

No LLM required for Pattern 2. This is pure data architecture: valuation generation, bridge decomposition, and governance dashboards.

All data is synthetic. No real fund names, NAVs, or counterparties.

## Setup

### 1. Databricks

Clone this repo into [Databricks Repos](https://docs.databricks.com/repos/index.html). The `.py` files render as notebooks.

### 2. Anthropic API key (Pattern 1 only)

Store your key in Databricks Secrets:

```bash
databricks secrets create-scope extraction-demo
databricks secrets put-secret extraction-demo anthropic-api-key
```

Notebooks read the key via:
```python
api_key = dbutils.secrets.get(scope="extraction-demo", key="anthropic-api-key")
```

Pattern 2 does not require an API key.

### 3. Unity Catalog

Notebooks write to `main.extraction_demo`. Adjust the catalog/schema widgets if your workspace uses different names.

### 4. Local (optional)

```bash
pip install anthropic reportlab matplotlib scikit-learn
```

Create `settings.local.json` in the project root (gitignored) with your API key:
```json
{"anthropic_api_key": "sk-ant-..."}
```

Pattern 1 (requires API key):
```bash
cd local
python 01_generate_synthetic_reports.py
python 02_extract_with_confidence.py --concurrency 8
python run_pipeline.py
```

Pattern 2 (no API key needed):
```bash
cd local
python run_pipeline.py --pattern 2
```

## Key design decisions

- **Synthetic data only**: no real GP reports or proprietary formats
- **PDF-native (Pattern 1)**: generated with reportlab, extracted with LLM vision
- **Prompt-based extraction**: no fine-tuning, keeps it reproducible
- **Field-level confidence**: the core architectural insight, not document-level
- **Source lineage**: every extracted value carries the verbatim source text, page number, and file path
- **Bridge decomposition (Pattern 2)**: operational, assumption, and methodology components must sum to total NAV change
- **Delta tables throughout**: all outputs in Unity Catalog

## Article series

This code accompanies a LinkedIn article series on investment data patterns. The series covers six recurring patterns that every alternatives data team encounters, from document extraction to entity resolution to time-series uncertainty.

## License

MIT
