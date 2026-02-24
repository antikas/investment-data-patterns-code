# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Extract with Confidence Scoring
# MAGIC
# MAGIC Schema-aware extraction using Claude API. For each synthetic GP report:
# MAGIC 1. Classify document type (quarterly report, capital call, distribution notice)
# MAGIC 2. Extract structured fields using schema-aware prompts per document type
# MAGIC 3. Capture field-level confidence scores (model self-assessment)
# MAGIC 4. Record lineage: source text, page, and file for every extracted value
# MAGIC
# MAGIC **Widgets:**
# MAGIC - `concurrency`: Number of parallel API calls (default 8; 1=sequential)
# MAGIC - `resume`: Skip documents already in the extractions table
# MAGIC
# MAGIC Output: Delta table `{catalog}.{schema}.extractions` with one row per field per report.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Check and install dependencies not available on the base runtime
import importlib.util, subprocess

_missing = [pkg for pkg, mod in [("anthropic", "anthropic"), ("pypdf", "pypdf"), ("nest_asyncio", "nest_asyncio")] if importlib.util.find_spec(mod) is None]
if _missing:
    subprocess.check_call(["pip", "install"] + _missing)
    dbutils.library.restartPython()

# COMMAND ----------

import anthropic
import asyncio
import json
import re
import time
from typing import Any

import nest_asyncio
nest_asyncio.apply()  # Allow asyncio.run() inside Databricks event loop

# Create widgets with defaults — run this cell first, then adjust values before running the next cell
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("schema", "extraction_demo", "Schema Name")
dbutils.widgets.text("concurrency", "8", "Parallel API calls (1=sequential, 5-10 for faster)")
dbutils.widgets.dropdown("resume", "false", ["true", "false"], "Resume (skip already extracted)")

# COMMAND ----------

# Read widget values — configure the widgets above before running this cell
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

api_key = dbutils.secrets.get(scope="extraction-demo", key="anthropic-api-key")
client = anthropic.Anthropic(api_key=api_key)
async_client = anthropic.AsyncAnthropic(api_key=api_key)

concurrency = int(dbutils.widgets.get("concurrency"))
resume_mode = dbutils.widgets.get("resume") == "true"
print(f"Mode: {'parallel (' + str(concurrency) + ' concurrent)' if concurrency > 1 else 'sequential'}")
if resume_mode:
    print("Resume: skipping already-extracted documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extraction Schema
# MAGIC
# MAGIC Define what fields we want to extract and their expected types.
# MAGIC This is the "schema-aware" part — the prompt tells Claude exactly what structure to produce.

# COMMAND ----------

QUARTERLY_REPORT_SCHEMA = {
    "fund_name": {"type": "string", "description": "Full legal name of the fund"},
    "gp_name": {"type": "string", "description": "Name of the General Partner / management company"},
    "reporting_period": {"type": "string", "description": "Reporting period, e.g. 'Q3 2025'"},
    "quarter_end_date": {"type": "string", "description": "Quarter end date in YYYY-MM-DD format"},
    "vintage_year": {"type": "integer", "description": "Year the fund began investing"},
    "strategy": {"type": "string", "description": "Investment strategy (e.g. Buyout, Growth Equity, Infrastructure)"},
    "currency": {"type": "string", "description": "Reporting currency (e.g. USD, EUR)"},
    "committed_capital_mm": {"type": "number", "description": "Total committed capital in millions"},
    "called_capital_mm": {"type": "number", "description": "Capital called to date in millions"},
    "distributed_capital_mm": {"type": "number", "description": "Capital distributed to date in millions"},
    "nav_mm": {"type": "number", "description": "Net Asset Value in millions"},
    "net_irr_pct": {"type": "number", "description": "Net IRR as a percentage (e.g. 14.3 for 14.3%)"},
    "tvpi": {"type": "number", "description": "Total Value to Paid-In multiple"},
    "dpi": {"type": "number", "description": "Distributions to Paid-In multiple"},
    "management_fee_mm": {"type": "number", "description": "Management fee for the period in millions"},
    "other_expenses_mm": {"type": "number", "description": "Other fund expenses for the period in millions"},
    "num_portfolio_companies": {"type": "integer", "description": "Number of portfolio companies mentioned"},
}

CAPITAL_CALL_SCHEMA = {
    "fund_name": {"type": "string", "description": "Full legal name of the fund"},
    "gp_name": {"type": "string", "description": "Name of the General Partner / management company"},
    "call_date": {"type": "string", "description": "Date of capital call notice in YYYY-MM-DD format"},
    "due_date": {"type": "string", "description": "Payment due date in YYYY-MM-DD format"},
    "call_amount_mm": {"type": "number", "description": "Amount being called in millions"},
    "call_amount_pct": {"type": "number", "description": "Call amount as percentage of total commitment"},
    "cumulative_called_mm": {"type": "number", "description": "Total capital called to date in millions"},
    "unfunded_commitment_mm": {"type": "number", "description": "Remaining unfunded commitment in millions"},
    "bank_name": {"type": "string", "description": "Name of the receiving bank"},
    "account_name": {"type": "string", "description": "Account name for wire transfers"},
    "account_number": {"type": "string", "description": "Bank account number"},
    "routing_number": {"type": "string", "description": "Bank routing number, sort code, or similar identifier"},
    "swift_code": {"type": "string", "description": "SWIFT/BIC code for international transfers"},
    "iban": {"type": "string", "description": "IBAN for European transfers"},
    "lp_commitment_reference": {"type": "string", "description": "LP's commitment reference number"},
    "vintage_year": {"type": "integer", "description": "Year the fund began investing"},
    "currency": {"type": "string", "description": "Currency of the call (e.g. USD, EUR)"},
}

DISTRIBUTION_SCHEMA = {
    "fund_name": {"type": "string", "description": "Full legal name of the fund"},
    "gp_name": {"type": "string", "description": "Name of the General Partner / management company"},
    "distribution_date": {"type": "string", "description": "Date of distribution in YYYY-MM-DD format"},
    "distribution_amount_mm": {"type": "number", "description": "Amount being distributed in millions"},
    "distribution_type": {"type": "string", "description": "Type of distribution: 'return_of_capital', 'income', or 'gain'"},
    "cumulative_distributed_mm": {"type": "number", "description": "Total distributions to date in millions"},
    "realization_source": {"type": "string", "description": "Source of the distribution (e.g. sale of portfolio company)"},
    "lp_commitment_reference": {"type": "string", "description": "LP's commitment reference number"},
    "vintage_year": {"type": "integer", "description": "Year the fund began investing"},
    "currency": {"type": "string", "description": "Currency of the distribution (e.g. USD, EUR)"},
}

# Schema lookup by document type
SCHEMAS = {
    "quarterly_report": QUARTERLY_REPORT_SCHEMA,
    "capital_call": CAPITAL_CALL_SCHEMA,
    "distribution": DISTRIBUTION_SCHEMA,
}

# Legacy schema name for backward compatibility
EXTRACTION_SCHEMA = QUARTERLY_REPORT_SCHEMA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extraction Prompt
# MAGIC
# MAGIC The prompt does three things:
# MAGIC 1. Extracts each field from the document
# MAGIC 2. Asks the model to self-assess confidence per field (0.0-1.0)
# MAGIC 3. Asks for the source text span that supports each extraction
# MAGIC
# MAGIC Model self-assessed confidence is a starting point, not the final score.
# MAGIC Notebook 03 calibrates these raw scores against empirical accuracy.

# COMMAND ----------

def classify_document_type(report_text: str) -> str:
    """Classify document as quarterly_report, capital_call, or distribution."""
    
    # Simple keyword-based classification  
    text_lower = report_text.lower()
    
    # Capital call indicators
    capital_call_keywords = [
        "capital call", "capital contribution", "wire transfer", 
        "payment due date", "account number", "routing number",
        "call notice", "capital required", "fund your commitment"
    ]
    
    # Distribution indicators  
    distribution_keywords = [
        "distribution notice", "distribution", "capital returned",
        "return of capital", "income distribution", "capital gain",
        "distribution date", "realization", "proceeds"
    ]
    
    # Quarterly report indicators
    quarterly_keywords = [
        "quarterly report", "quarter end", "nav", "irr", 
        "portfolio companies", "tvpi", "dpi", "performance metrics"
    ]
    
    # Score each type
    capital_score = sum(1 for kw in capital_call_keywords if kw in text_lower)
    distribution_score = sum(1 for kw in distribution_keywords if kw in text_lower)  
    quarterly_score = sum(1 for kw in quarterly_keywords if kw in text_lower)
    
    # Determine document type
    if capital_score > distribution_score and capital_score > quarterly_score:
        return "capital_call"
    elif distribution_score > quarterly_score:
        return "distribution" 
    else:
        return "quarterly_report"


def build_extraction_prompt(report_text: str, source_filename: str = "document.pdf") -> str:
    """Build extraction prompt with document type classification and appropriate schema."""
    # First classify the document
    doc_type = classify_document_type(report_text)
    schema = SCHEMAS[doc_type]
    
    schema_description = "\n".join(
        f"  - {field}: ({spec['type']}) {spec['description']}"
        for field, spec in schema.items()
    )

    return f"""You are extracting structured data from a GP document ({doc_type.replace('_', ' ')}).

Extract the following fields from the document below. For each field, provide:
1. "value": the extracted value (use null if the field is not present or cannot be determined)
2. "confidence": your confidence in the extraction, from 0.0 to 1.0
   - 1.0: the value is explicitly and unambiguously stated
   - 0.7-0.9: the value is present but requires interpretation (e.g. different units, embedded in prose)
   - 0.4-0.6: the value is inferred or partially present
   - 0.0-0.3: the value is guessed or very uncertain
3. "source_text": the exact text span from the document that supports this extraction (verbatim quote, or null if not found)
4. "source_page": page number where the source text was found (use 1 if single-page or unknown)
5. "source_file": the source filename ("{source_filename}")

CRITICAL: Source attribution is required for audit compliance.
- Source text must be sufficient for human verification of the extraction
- Prefer lower confidence with good attribution over high confidence without source

Fields to extract:
{schema_description}

Example response format:
{{
  "fund_name": {{
    "value": "Example Fund III",
    "confidence": 0.95,
    "source_text": "Example Fund III Quarterly Report for the Period Ending December 31, 2024",
    "source_page": 1,
    "source_file": "{source_filename}"
  }},
  "missing_field": {{
    "value": null,
    "confidence": 0.0,
    "source_text": null,
    "source_page": null,
    "source_file": "{source_filename}"
  }}
}}

DOCUMENT:
---
{report_text}
---

Extract all fields with complete source lineage. Return only valid JSON, no other text."""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Extraction

# COMMAND ----------

def _try_parse_json(text: str) -> dict | None:
    """Try to parse JSON, applying common LLM output repairs if needed."""
    # 1. Try as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fix trailing commas before } or ] (most common LLM error)
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 3. Truncated response — try closing open braces/brackets
    bracket_depth = text.count("{") - text.count("}")
    square_depth = text.count("[") - text.count("]")
    if bracket_depth > 0 or square_depth > 0:
        closed = text.rstrip().rstrip(",")
        closed += "]" * square_depth + "}" * bracket_depth
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            pass

    return None


def parse_extraction_response(response_text: str, fund_id: str = "") -> dict:
    """Parse JSON from Claude API response, handling markdown code blocks and common LLM errors."""
    # Try raw response
    result = _try_parse_json(response_text)
    if result is not None:
        return result

    # Try extracting from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        result = _try_parse_json(json_match.group(1))
        if result is not None:
            return result

    # Try greedy extraction (largest {...} block)
    json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if json_match:
        result = _try_parse_json(json_match.group(1))
        if result is not None:
            return result

    if fund_id:
        print(f"  WARNING: Failed to parse JSON for {fund_id} — skipping document")
    return {}


MAX_RETRIES = 3
RETRY_BACKOFF = [10, 30, 60]


def extract_report(report_text: str, fund_id: str, source_filename: str) -> dict:
    """Extract fields from a single report (synchronous, with retry)."""
    prompt = build_extraction_prompt(report_text, source_filename)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=6144,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_extraction_response(response.content[0].text, fund_id)
        except Exception as e:
            err_str = str(e)
            is_transient = "overloaded" in err_str.lower() or "529" in err_str or "429" in err_str
            if is_transient and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                print(f"  [retry {attempt+1}/{MAX_RETRIES} in {wait}s: {err_str[:60]}]")
                time.sleep(wait)
            else:
                raise
    return {}


async def extract_report_async(report_text: str, fund_id: str, source_filename: str) -> dict:
    """Extract fields from a single report (async, with retry)."""
    prompt = build_extraction_prompt(report_text, source_filename)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await async_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=6144,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_extraction_response(response.content[0].text, fund_id)
        except Exception as e:
            err_str = str(e)
            is_transient = "overloaded" in err_str.lower() or "529" in err_str or "429" in err_str
            if is_transient and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                print(f"  [retry {attempt+1}/{MAX_RETRIES} in {wait}s: {err_str[:60]}]")
                await asyncio.sleep(wait)
            else:
                raise
    return {}

# COMMAND ----------

# Load report manifest — PDFs are stored in the Unity Catalog Volume
from pypdf import PdfReader

reports_df = spark.sql("SELECT fund_id, fund_name, reporting_period, report_quality_tier, source_document_id, path FROM report_manifest")
reports = reports_df.collect()

# Resume mode: skip already-extracted documents (by source_document_id, not fund_id)
already_extracted = set()
if resume_mode:
    try:
        existing = spark.sql(f"SELECT DISTINCT source_file FROM {catalog}.{schema}.extractions").collect()
        already_extracted = {row.source_file for row in existing}
        print(f"Resume: {len(already_extracted)} documents already extracted, skipping")
    except Exception:
        print("No existing extractions table found, extracting all")

reports_to_extract = [r for r in reports if r.source_document_id not in already_extracted]
print(f"Extracting from {len(reports_to_extract)} reports (of {len(reports)} total)...")

# COMMAND ----------

def flatten_extraction(fund_id, tier, report_text, extraction):
    """Flatten extraction result to one-row-per-field format for Delta table."""
    doc_type = classify_document_type(report_text)
    schema = SCHEMAS[doc_type]
    rows = []
    for field_name, result in extraction.items():
        if field_name not in schema:
            continue
        rows.append({
            "fund_id": fund_id,
            "report_quality_tier": tier,
            "document_type": doc_type,
            "field_name": field_name,
            "extracted_value": str(result.get("value")) if result.get("value") is not None else None,
            "raw_confidence": float(result.get("confidence", 0.0)),
            "source_text": result.get("source_text"),
            "source_page": result.get("source_page"),
            "source_file": result.get("source_file"),
            "field_type": schema[field_name]["type"],
        })
    return rows, doc_type


# Run extraction — sequential or parallel based on concurrency widget
all_extractions = []

if concurrency <= 1:
    # Sequential extraction
    for i, row in enumerate(reports_to_extract):
        reader = PdfReader(row.path)
        report_text = "\n".join(
            f"=== PAGE {p+1} ===\n{page.extract_text() or ''}"
            for p, page in enumerate(reader.pages)
        )

        print(f"[{i+1}/{len(reports_to_extract)}] {row.fund_id} [{row.report_quality_tier}]", end=" ")
        source_filename = row.path.split("/")[-1] if "/" in row.path else row.path

        t0 = time.time()
        extraction = extract_report(report_text, row.fund_id, source_filename)
        elapsed = time.time() - t0

        rows, doc_type = flatten_extraction(row.fund_id, row.report_quality_tier, report_text, extraction)
        all_extractions.extend(rows)

        field_count = len(rows)
        avg_conf = sum(r["raw_confidence"] for r in rows) / max(field_count, 1)
        print(f"-> {doc_type}, {field_count} fields, avg conf {avg_conf:.2f}, {elapsed:.1f}s")

else:
    # Parallel extraction with throttled concurrency
    semaphore = asyncio.Semaphore(concurrency)
    completed = [0]
    lock = asyncio.Lock()
    total = len(reports_to_extract)
    t_start = time.time()

    async def process_one(row):
        reader = PdfReader(row.path)
        report_text = "\n".join(
            f"=== PAGE {p+1} ===\n{page.extract_text() or ''}"
            for p, page in enumerate(reader.pages)
        )
        source_filename = row.path.split("/")[-1] if "/" in row.path else row.path

        async with semaphore:
            t0 = time.time()
            try:
                extraction = await extract_report_async(report_text, row.fund_id, source_filename)
            except Exception as e:
                async with lock:
                    completed[0] += 1
                    print(f"[{completed[0]}/{total}] {row.fund_id} ERROR: {e}")
                return []
            elapsed = time.time() - t0

        rows, doc_type = flatten_extraction(row.fund_id, row.report_quality_tier, report_text, extraction)

        async with lock:
            completed[0] += 1
            elapsed_total = time.time() - t_start
            rate = completed[0] / elapsed_total if elapsed_total > 0 else 0
            field_count = len(rows)
            avg_conf = sum(r["raw_confidence"] for r in rows) / max(field_count, 1)
            print(
                f"[{completed[0]}/{total}] {row.fund_id} [{row.report_quality_tier}] "
                f"-> {doc_type}, {field_count} fields, avg conf {avg_conf:.2f}, "
                f"{elapsed:.1f}s ({rate:.1f} docs/s)"
            )
        return rows

    async def run_all():
        tasks = [asyncio.create_task(process_one(row)) for row in reports_to_extract]
        results = await asyncio.gather(*tasks)
        for rows in results:
            all_extractions.extend(rows)

    asyncio.run(run_all())

print(f"\nExtraction complete: {len(all_extractions)} field-level rows from {len(reports_to_extract)} documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Extractions to Delta

# COMMAND ----------

extractions_df = spark.createDataFrame(all_extractions)

if resume_mode and already_extracted:
    # Append new extractions to existing table
    extractions_df.write.mode("append").saveAsTable(f"{catalog}.{schema}.extractions")
    total_count = spark.sql(f"SELECT COUNT(*) as cnt FROM {catalog}.{schema}.extractions").collect()[0].cnt
    print(f"Appended {extractions_df.count()} rows ({total_count} total) to {catalog}.{schema}.extractions")
else:
    extractions_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.extractions")
    print(f"Wrote {extractions_df.count()} field extractions to {catalog}.{schema}.extractions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Analysis
# MAGIC
# MAGIC Before we get to calibration (Notebook 03), a first look at what we got.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Average raw confidence by quality tier
# MAGIC SELECT
# MAGIC   report_quality_tier,
# MAGIC   COUNT(*) as num_fields,
# MAGIC   ROUND(AVG(raw_confidence), 3) as avg_confidence,
# MAGIC   ROUND(MIN(raw_confidence), 3) as min_confidence,
# MAGIC   ROUND(MAX(raw_confidence), 3) as max_confidence
# MAGIC FROM extractions
# MAGIC GROUP BY report_quality_tier
# MAGIC ORDER BY avg_confidence DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Which fields have lowest confidence across all reports?
# MAGIC SELECT
# MAGIC   field_name,
# MAGIC   report_quality_tier,
# MAGIC   ROUND(AVG(raw_confidence), 3) as avg_confidence,
# MAGIC   COUNT(CASE WHEN extracted_value IS NULL THEN 1 END) as null_count
# MAGIC FROM extractions
# MAGIC GROUP BY field_name, report_quality_tier
# MAGIC ORDER BY avg_confidence ASC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Distribution of confidence scores
# MAGIC SELECT
# MAGIC   CASE
# MAGIC     WHEN raw_confidence >= 0.9 THEN '0.9-1.0 (high)'
# MAGIC     WHEN raw_confidence >= 0.7 THEN '0.7-0.9 (medium)'
# MAGIC     WHEN raw_confidence >= 0.4 THEN '0.4-0.7 (low)'
# MAGIC     ELSE '0.0-0.4 (very low)'
# MAGIC   END as confidence_band,
# MAGIC   COUNT(*) as field_count,
# MAGIC   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as pct
# MAGIC FROM extractions
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We now have:
# MAGIC - **Extracted values** for every field across all reports (~27 funds x 7 quarters + capital calls + distributions)
# MAGIC - **Raw confidence scores** (model self-assessed) per field
# MAGIC - **Source text spans** linking each extraction to the source document
# MAGIC
# MAGIC The raw confidence scores are the model's self-assessment. They are NOT calibrated.
# MAGIC A confidence of 0.9 does not yet mean "correct 90% of the time."
# MAGIC
# MAGIC Next: **Notebook 03** — Compare extractions against ground truth, calibrate confidence
# MAGIC scores to empirical accuracy, and set routing thresholds.
