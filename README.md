# Investment Data Patterns: Extraction Pipeline Demo

Companion code for the article series on recurring data patterns in alternative investments.

This repo demonstrates **Pattern 1: Structured-from-Unstructured**. LLM-powered document extraction with field-level confidence scoring, isotonic calibration, routing thresholds, and full source lineage. The pipeline processes synthetic GP quarterly reports, capital call notices, and distribution notices across three quality tiers.

## Why field-level confidence matters

Most extraction pipelines return a document-level confidence score. That's not useful when your NAV is extracted at 0.97 confidence but the IRR on the same page is 0.42. The same problem shows up across document types: a capital call's total amount might be unambiguous while the allocation breakdown is buried in a footnote, or a distribution notice's payment date is clear but the return-of-capital vs income split requires interpretation. Investment teams need to know which *fields* to trust and which to route for human review. Per document type, per field, per quality tier.

This demo builds that infrastructure: raw confidence from the LLM, isotonic calibration against ground truth, threshold-based routing (auto-accept vs human review), and full lineage from extracted value back to source text and page.

## What's here

### Pipeline (runs end-to-end on Databricks)

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_cleanup.py` | Drops all tables and files. Idempotent, safe to run at any point. |
| 01 | `01_generate_synthetic_reports.py` | Generates ~450 synthetic PDF documents (quarterly reports, capital calls, distributions) across 27 funds, 7 quarters, 3 quality tiers. Writes ground truth to Delta tables. |
| 02 | `02_extract_with_confidence.py` | PDF to LLM vision extraction with field-level confidence and source lineage. Supports parallel extraction and resume mode. |
| 03 | `03_calibration_and_routing.py` | Isotonic regression calibration, routing threshold trade-off analysis, review queue simulation. |
| 04 | `04_cost_model.py` | Rules-based vs LLM extraction economics, parameterised crossover model, 3-year TCO. |
| 05 | `05_portfolio_dashboard.py` | Dataset exploration queries for the Lakeview portfolio dashboard. |
| 06 | `06_pipeline_metrics.lvdash.json` | Lakeview dashboard: extraction accuracy, calibration curves, routing decisions, review queue, source lineage, cost model. |

### Local scripts (development / no cluster needed)

| File | What it does |
|------|--------------|
| `01_generate_synthetic_reports.py` | Same generation logic, outputs PDFs + JSON locally for visual inspection |
| `02_extract_with_confidence.py` | LLM API extraction with `--concurrency N`, `--resume`, `--sample N` |
| `03_calibration_and_routing.py` | Per-document-type isotonic calibration and routing threshold sweep |
| `04_cost_model.py` | Rules-based vs LLM cost model with multi-document economics |
| `05_portfolio_dashboard.py` | Self-contained HTML dashboard with embedded charts |
| `run_pipeline.py` | Runs 03→04→05 from stored extraction results. Use `--from 04` to skip steps. |

### Shared modules

All domain logic lives in `shared/`, imported by both local scripts and Databricks notebooks.

| Module | Purpose |
|--------|---------|
| `fund_definitions.py` | 27-fund universe: strategies, vintages, committed capital, quality tiers |
| `simulation.py` | Deterministic fund performance simulation (J-curve, distributions, IRR) |
| `report_generators.py` | PDF generation with reportlab (3 quality tiers × 3 document types) |
| `notice_generators.py` | Capital call and distribution notice PDF generation |
| `failure_modes.py` | The 7 GP report failure modes: format, timing, terminology, currency, restatements, transcription errors, number ambiguity |
| `schemas.py` | Extraction schemas per document type (SSOT for prompts and accuracy measurement) |
| `extraction_utils.py` | Flat row → nested extraction dict reconstruction |

## Synthetic dataset

**27 funds** across 7 strategies (buyout, growth equity, infrastructure, real estate, private credit, secondaries, venture capital). **7 quarters** (Q1 2024 – Q3 2025). **3 quality tiers** reflecting the real-world reporting spectrum:

- **Institutional**: Clean tabular layout, consistent fields (large GP)
- **Narrative**: Numbers embedded in quarterly letter prose (mid-market GP)
- **Poor quality**: Sparse, inconsistent structure (small manager)

Three document types per fund per quarter: quarterly reports, capital call notices, and distribution notices. The dataset includes the 7 common GP report failure modes: format variation, timing variation, terminology variation, currency variation, silent restatements, transcription errors, and number ambiguity.

All data is synthetic. No real fund names, NAVs, or counterparties.

## Setup

### 1. Databricks

Clone this repo into [Databricks Repos](https://docs.databricks.com/repos/index.html). The `.py` files render as notebooks.

### 2. Anthropic API key

Store your key in Databricks Secrets:

```bash
databricks secrets create-scope extraction-demo
databricks secrets put-secret extraction-demo anthropic-api-key
```

Notebooks read the key via:
```python
api_key = dbutils.secrets.get(scope="extraction-demo", key="anthropic-api-key")
```

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

Then run:
```bash
cd local
python 01_generate_synthetic_reports.py
python 02_extract_with_confidence.py --concurrency 8
python run_pipeline.py
```

## Key design decisions

- **Synthetic data only**: no real GP reports or proprietary formats
- **PDF-native**: generated with reportlab, extracted with LLM vision. Full pipeline: PDF generation, PDF-to-image, LLM extraction.
- **Prompt-based extraction**: no fine-tuning, keeps it reproducible
- **Field-level confidence**: the core architectural insight, not document-level
- **Source lineage**: every extracted value carries the verbatim source text, page number, and file path
- **Delta tables throughout**: extractions, ground truth, calibration, cost model all in Unity Catalog

## Article series

This code accompanies a LinkedIn article series on investment data patterns. The series covers six recurring patterns that every alternatives data team encounters, from document extraction to entity resolution to time-series uncertainty.

## License

MIT
