"""
02 -- Extract with Confidence Scoring and Source Lineage Tracking (Local)

Reads PDFs from local/output/reports/, extracts text, calls Claude API for
schema-aware extraction with field-level confidence scores and full source attribution.

KEY FEATURES:
- Document type classification (quarterly reports, capital calls, distribution notices)
- Schema-aware extraction with appropriate fields per document type
- Complete source lineage: source_text, source_page, source_file for every field
- Confidence scoring with bias toward lower confidence for fields without source attribution
- Audit trail: every extracted value traceable back to exact source text and location
- Parallel extraction with configurable concurrency (--concurrency N)

Usage:
    python local/02_extract_with_confidence.py                  # all reports (sequential)
    python local/02_extract_with_confidence.py --concurrency 5  # 5 parallel API calls
    python local/02_extract_with_confidence.py --sample 5       # 5 random reports
    python local/02_extract_with_confidence.py --resume         # skip already-extracted

Requires:
    pip install pypdf anthropic
    Anthropic API key in settings.local.json (gitignored)
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from pypdf import PdfReader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "local" / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
EXTRACTIONS_PATH = OUTPUT_DIR / "extractions.json"

# ---------------------------------------------------------------------------
# Extraction schemas for different document types
# ---------------------------------------------------------------------------

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

# Legacy schema for backward compatibility
EXTRACTION_SCHEMA = QUARTERLY_REPORT_SCHEMA


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


def build_extraction_prompt(report_text: str, doc_type: str, source_filename: str) -> str:
    schema = SCHEMAS[doc_type]
    schema_description = "\n".join(
        f"  - {field}: ({spec['type']}) {spec['description']}"
        for field, spec in schema.items()
    )

    doc_type_names = {
        "quarterly_report": "GP quarterly report",
        "capital_call": "capital call notice",
        "distribution": "distribution notice"
    }
    doc_name = doc_type_names[doc_type]

    return f"""You are extracting structured data from a {doc_name} with full source lineage tracking.

CRITICAL REQUIREMENT: Every extracted field must include complete source attribution for audit trail and trust.

Extract the following fields from the document below. For each field, provide:

1. "value": the extracted value (use null if not present/determinable)

2. "confidence": your extraction confidence from 0.0 to 1.0
   - 1.0: explicitly and unambiguously stated with clear source text
   - 0.7-0.9: present but requires interpretation, with identifiable source
   - 0.4-0.6: inferred or partially present, with weak source attribution
   - 0.0-0.3: guessed, very uncertain, or NO source text found
   - NOTE: Fields without clear source_text should automatically receive confidence <= 0.3

3. "source_text": EXACT verbatim quote from the document supporting this extraction
   - Must be word-for-word identical to document text
   - Include sufficient context (1-2 sentences) to verify the extraction
   - Use null only if the field is completely absent from the document

4. "source_page": Page number where the source_text was found (look for "=== PAGE N ===" markers)
   - Use integer page number (1, 2, 3, etc.)
   - Use null only if source_text is null

5. "source_file": Source filename (use: "{source_filename}")

Fields to extract:
{schema_description}

QUALITY CONTROL RULES:
- If you cannot find verbatim source text for a value, set confidence <= 0.3
- Every non-null extraction must have corresponding source_text and source_page
- Source text must be sufficient for human verification of the extraction
- Prefer lower confidence with good attribution over high confidence without source

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


def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict]:
    """Extract text from PDF and return both full text and page-by-page mapping."""
    reader = PdfReader(pdf_path)
    pages_text = {}
    full_text_parts = []

    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        pages_text[page_num] = page_text
        full_text_parts.append(f"=== PAGE {page_num} ===\n{page_text}")

    full_text = "\n".join(full_text_parts)
    return full_text, pages_text


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


def parse_extraction_response(response_text: str) -> dict:
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

    return {}


def compute_extraction_stats(extraction: dict, schema: dict) -> tuple[int, float, float]:
    """Compute stats for a single extraction. Returns (fields_extracted, avg_conf, source_pct)."""
    fields_extracted = 0
    fields_with_source = 0
    total_conf = 0

    for field_name, field_result in extraction.items():
        if field_name not in schema:
            continue
        fields_extracted += 1
        total_conf += field_result.get("confidence", 0)
        has_source = (
            field_result.get("source_text") is not None and
            field_result.get("source_page") is not None and
            field_result.get("source_file") is not None
        )
        if has_source:
            fields_with_source += 1

    avg_conf = total_conf / fields_extracted if fields_extracted > 0 else 0
    source_pct = (fields_with_source / fields_extracted * 100) if fields_extracted > 0 else 0
    return fields_extracted, avg_conf, source_pct


def build_extraction_record(report: dict, doc_type: str, extraction: dict) -> dict:
    """Build the extraction record dict from report manifest entry + extraction result."""
    return {
        "source_document_id": report["source_document_id"],
        "fund_id": report["fund_id"],
        "report_quality_tier": report["report_quality_tier"],
        "document_type": doc_type,
        "reporting_period": report.get("reporting_period"),
        "call_date": report.get("call_date"),
        "distribution_date": report.get("distribution_date"),
        "source_file": report["path"],
        "extraction": extraction,
    }


MAX_RETRIES = 3
RETRY_BACKOFF = [10, 30, 60]  # seconds between retries


def extract_report(client, report_text: str, source_filename: str, model: str) -> dict:
    """Extract fields from a single report via Claude API (synchronous, with retry)."""
    doc_type = classify_document_type(report_text)
    prompt = build_extraction_prompt(report_text, doc_type, source_filename)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=6144,
                messages=[{"role": "user", "content": prompt}],
            )
            extraction = parse_extraction_response(response.content[0].text)
            return {"document_type": doc_type, "extraction": extraction}
        except Exception as e:
            err_str = str(e)
            is_transient = "overloaded" in err_str.lower() or "529" in err_str or "429" in err_str
            if is_transient and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                print(f"  [retry {attempt+1}/{MAX_RETRIES} in {wait}s: {err_str[:60]}]", flush=True)
                time.sleep(wait)
            else:
                raise

    return {"document_type": doc_type, "extraction": {}}


async def extract_report_async(async_client, report_text: str, source_filename: str, model: str) -> dict:
    """Extract fields from a single report via Claude API (async, with retry)."""
    doc_type = classify_document_type(report_text)
    prompt = build_extraction_prompt(report_text, doc_type, source_filename)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await async_client.messages.create(
                model=model,
                max_tokens=6144,
                messages=[{"role": "user", "content": prompt}],
            )
            extraction = parse_extraction_response(response.content[0].text)
            return {"document_type": doc_type, "extraction": extraction}
        except Exception as e:
            err_str = str(e)
            is_transient = "overloaded" in err_str.lower() or "529" in err_str or "429" in err_str
            if is_transient and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                print(f"  [retry {attempt+1}/{MAX_RETRIES} in {wait}s: {err_str[:60]}]", flush=True)
                await asyncio.sleep(wait)
            else:
                raise

    return {"document_type": doc_type, "extraction": {}}


def save_extractions(all_extractions: list):
    """Write extractions to JSON (crash-safe)."""
    with open(EXTRACTIONS_PATH, "w") as f:
        json.dump(all_extractions, f, indent=2)


def print_extraction_summary(all_extractions: list):
    """Print final summary of all extractions."""
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE WITH SOURCE LINEAGE TRACKING")
    print(f"{'='*70}")
    print(f"  Reports extracted: {len(all_extractions)}")
    print(f"  Output: {EXTRACTIONS_PATH}")
    print(f"  Source attribution: Every field includes source_text, source_page, source_file")
    print(f"  Audit trail: Complete traceability from extracted value to source document")

    by_tier = {}
    by_doc_type = {}

    for ext in all_extractions:
        tier = ext["report_quality_tier"]
        doc_type = ext.get("document_type", "quarterly_report")

        if tier not in by_tier:
            by_tier[tier] = {"count": 0, "total_conf": 0, "total_fields": 0, "sourced_fields": 0}
        by_tier[tier]["count"] += 1

        if doc_type not in by_doc_type:
            by_doc_type[doc_type] = {"count": 0, "total_conf": 0, "total_fields": 0, "sourced_fields": 0}
        by_doc_type[doc_type]["count"] += 1

        schema = SCHEMAS.get(doc_type, QUARTERLY_REPORT_SCHEMA)
        for field_name, result in ext["extraction"].items():
            if field_name in schema:
                conf = result.get("confidence", 0)
                by_tier[tier]["total_conf"] += conf
                by_tier[tier]["total_fields"] += 1
                by_doc_type[doc_type]["total_conf"] += conf
                by_doc_type[doc_type]["total_fields"] += 1

                has_source = (
                    result.get("source_text") is not None and
                    result.get("source_page") is not None and
                    result.get("source_file") is not None
                )
                if has_source:
                    by_tier[tier]["sourced_fields"] += 1
                    by_doc_type[doc_type]["sourced_fields"] += 1

    print(f"\n  {'Tier':<15} {'Reports':>8} {'Avg Conf':>10} {'Source %':>10}")
    print(f"  {'-'*45}")
    for tier in ["institutional", "narrative", "poor"]:
        if tier in by_tier:
            t = by_tier[tier]
            avg_conf = t["total_conf"] / t["total_fields"] if t["total_fields"] > 0 else 0
            source_pct = (t["sourced_fields"] / t["total_fields"] * 100) if t["total_fields"] > 0 else 0
            print(f"  {tier:<15} {t['count']:>8} {avg_conf:>10.3f} {source_pct:>9.0f}%")

    print(f"\n  {'Document Type':<18} {'Count':>8} {'Avg Conf':>10} {'Source %':>10}")
    print(f"  {'-'*48}")
    for doc_type in ["quarterly_report", "capital_call", "distribution"]:
        if doc_type in by_doc_type:
            t = by_doc_type[doc_type]
            avg_conf = t["total_conf"] / t["total_fields"] if t["total_fields"] > 0 else 0
            source_pct = (t["sourced_fields"] / t["total_fields"] * 100) if t["total_fields"] > 0 else 0
            display_name = doc_type.replace("_", " ").title()
            print(f"  {display_name:<18} {t['count']:>8} {avg_conf:>10.3f} {source_pct:>9.0f}%")

    total_fields = sum(t["total_fields"] for t in by_doc_type.values())
    total_sourced = sum(t["sourced_fields"] for t in by_doc_type.values())
    overall_source_pct = (total_sourced / total_fields * 100) if total_fields > 0 else 0

    print(f"\n  {'SOURCE LINEAGE SUMMARY':<30}")
    print(f"  {'-'*48}")
    print(f"  Total extracted fields: {total_fields}")
    print(f"  Fields with source attribution: {total_sourced} ({overall_source_pct:.1f}%)")
    print(f"  Fields with source text, page, and file references")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Sequential extraction (default, --concurrency 1)
# ---------------------------------------------------------------------------

def run_sequential(reports, client, model, existing_extractions):
    """Run extraction sequentially, one report at a time."""
    all_extractions = list(existing_extractions)

    for i, report in enumerate(reports):
        doc_id = report["source_document_id"]
        tier = report["report_quality_tier"]

        print(f"[{i+1}/{len(reports)}] {doc_id} [{tier}]", end=" ", flush=True)

        report_text, _ = extract_text_from_pdf(report["path"])
        if not report_text.strip():
            print("SKIPPED (empty text)")
            continue

        t0 = time.time()
        result = extract_report(client, report_text, doc_id, model)
        elapsed = time.time() - t0

        doc_type = result["document_type"]
        extraction = result["extraction"]
        schema = SCHEMAS.get(doc_type, QUARTERLY_REPORT_SCHEMA)
        fields, avg_conf, source_pct = compute_extraction_stats(extraction, schema)

        print(f"-> {doc_type}, {fields} fields, avg conf {avg_conf:.2f}, {source_pct:.0f}% sourced, {elapsed:.1f}s")

        all_extractions.append(build_extraction_record(report, doc_type, extraction))
        save_extractions(all_extractions)

    return all_extractions


# ---------------------------------------------------------------------------
# Parallel extraction (--concurrency N where N > 1)
# ---------------------------------------------------------------------------

async def run_parallel(reports, api_key, model, concurrency, existing_extractions):
    """Run extraction with N concurrent API calls, throttled by semaphore."""
    import anthropic

    async_client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    all_extractions = list(existing_extractions)
    completed = 0
    failed = 0
    lock = asyncio.Lock()
    total = len(reports)
    t_start = time.time()

    async def process_one(report):
        nonlocal completed, failed

        doc_id = report["source_document_id"]
        tier = report["report_quality_tier"]

        # PDF text extraction (fast, synchronous)
        report_text, _ = extract_text_from_pdf(report["path"])
        if not report_text.strip():
            async with lock:
                completed += 1
                print(f"[{completed}/{total}] {doc_id} [{tier}] SKIPPED (empty text)")
            return

        # API call (throttled by semaphore)
        async with semaphore:
            t0 = time.time()
            try:
                result = await extract_report_async(async_client, report_text, doc_id, model)
            except Exception as e:
                async with lock:
                    completed += 1
                    failed += 1
                    print(f"[{completed}/{total}] {doc_id} [{tier}] ERROR: {e}")
                return
            elapsed = time.time() - t0

        doc_type = result["document_type"]
        extraction = result["extraction"]
        schema = SCHEMAS.get(doc_type, QUARTERLY_REPORT_SCHEMA)
        fields, avg_conf, source_pct = compute_extraction_stats(extraction, schema)

        record = build_extraction_record(report, doc_type, extraction)

        async with lock:
            all_extractions.append(record)
            completed += 1
            elapsed_total = time.time() - t_start
            rate = completed / elapsed_total if elapsed_total > 0 else 0
            print(
                f"[{completed}/{total}] {doc_id} [{tier}] "
                f"-> {doc_type}, {fields} fields, avg conf {avg_conf:.2f}, "
                f"{source_pct:.0f}% sourced, {elapsed:.1f}s "
                f"({rate:.1f} docs/s)"
            )
            save_extractions(all_extractions)

    # Launch all tasks - semaphore gates actual concurrency
    tasks = [asyncio.create_task(process_one(report)) for report in reports]
    await asyncio.gather(*tasks)

    if failed > 0:
        print(f"\nWarning: {failed} extractions failed (see ERROR lines above)")

    return all_extractions


def main():
    parser = argparse.ArgumentParser(description="Extract fields from GP reports using Claude API")
    parser.add_argument("--sample", type=int, default=0, help="Extract from N random reports (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip reports already in extractions.json")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of parallel API calls (default 1 = sequential, try 5-10 for faster runs)"
    )
    args = parser.parse_args()

    # Load API key from settings.local.json (project root)
    settings_path = PROJECT_ROOT / "settings.local.json"
    if not settings_path.exists():
        print(f"Error: {settings_path} not found.")
        print("Create it with: {\"anthropic_api_key\": \"sk-ant-...\"}")
        sys.exit(1)

    with open(settings_path) as f:
        settings = json.load(f)
    api_key = settings.get("anthropic_api_key", "")
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        print(f"Error: Set your Anthropic API key in {settings_path}")
        sys.exit(1)

    # Load manifest
    if not MANIFEST_PATH.exists():
        print("Error: manifest.json not found. Run local/01_generate_synthetic_reports.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Load existing extractions for resume mode
    existing = {}
    if args.resume and EXTRACTIONS_PATH.exists():
        with open(EXTRACTIONS_PATH) as f:
            existing_list = json.load(f)
        for ext in existing_list:
            existing[ext["source_document_id"]] = ext
        print(f"Resuming: {len(existing)} reports already extracted")

    # Filter/sample
    reports = manifest
    if args.resume:
        reports = [r for r in reports if r["source_document_id"] not in existing]
    if args.sample > 0:
        import random
        random.seed(42)
        reports = random.sample(reports, min(args.sample, len(reports)))

    if not reports:
        print("No reports to extract (all already done or none match filter)")
        if existing:
            print_extraction_summary(list(existing.values()))
        return

    concurrency = max(1, args.concurrency)
    mode = f"parallel ({concurrency} concurrent)" if concurrency > 1 else "sequential"
    print(f"Extracting from {len(reports)} reports (model: {args.model}, mode: {mode})")
    print(f"Output: {EXTRACTIONS_PATH}\n")

    t_start = time.time()

    if concurrency > 1:
        all_extractions = asyncio.run(
            run_parallel(reports, api_key, args.model, concurrency, existing.values())
        )
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        all_extractions = run_sequential(reports, client, args.model, existing.values())

    elapsed_total = time.time() - t_start
    docs_extracted = len(all_extractions) - len(existing)
    rate = docs_extracted / elapsed_total if elapsed_total > 0 else 0
    print(f"\nTotal time: {elapsed_total:.0f}s for {docs_extracted} documents ({rate:.1f} docs/s)")

    print_extraction_summary(all_extractions)


if __name__ == "__main__":
    main()
