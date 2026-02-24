"""
Capital call and distribution notice PDF generators.

Extends the existing report generation pattern to handle two new document types:
- Capital call notices: Fund requests capital from LPs
- Distribution notices: Fund returns capital/income to LPs

Each document type has quality tiers (institutional/narrative/poor) following 
the same pattern as quarterly reports. Key extraction challenges:
- Bank details in varying formats
- Call amounts as percentages vs absolute numbers 
- Distribution type classification (return of capital vs income)
- Document type classification (what kind of document is this?)
"""

import random
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)

from shared.fund_definitions import (
    FundDefinition,
    TERMINOLOGY_SETS,
    CURRENCY_SYMBOLS,
    GP_ADDRESSES,
)

# Use the same color scheme as quarterly reports
NAVY = HexColor("#1B2A4A")
DARK_GREY = HexColor("#333333")
MID_GREY = HexColor("#666666")
TABLE_HEADER_BG = HexColor("#1B2A4A")
TABLE_ALT_ROW = HexColor("#F0F4F8")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CapitalCallNotice:
    """Data structure for a capital call notice."""
    fund_id: str
    fund_name: str
    gp_name: str
    vintage_year: int
    strategy: str
    currency: str
    committed_capital_mm: float
    lp_commitment_mm: float
    
    # Capital call specific fields
    call_date: str              # Date of the call notice
    due_date: str               # Payment due date
    call_amount_mm: float       # Amount being called this time
    call_amount_pct: float      # As percentage of commitment
    cumulative_called_mm: float # Total called to date
    unfunded_commitment_mm: float # Remaining unfunded
    
    # Bank details
    bank_name: str
    account_name: str
    account_number: str
    routing_number: Optional[str]
    swift_code: Optional[str]
    iban: Optional[str]
    
    # LP reference
    lp_commitment_reference: str
    
    # Document metadata
    report_quality_tier: str
    source_document_id: str
    terminology: str

@dataclass 
class DistributionNotice:
    """Data structure for a distribution notice."""
    fund_id: str
    fund_name: str
    gp_name: str
    vintage_year: int
    strategy: str
    currency: str
    committed_capital_mm: float
    lp_commitment_mm: float
    
    # Distribution specific fields
    distribution_date: str       # Date of distribution
    distribution_amount_mm: float # Amount being distributed
    distribution_type: str       # "return_of_capital", "income", "gain"
    cumulative_distributed_mm: float # Total distributed to date
    
    # Source of distribution
    realization_source: Optional[str] # e.g. "sale of Meridian Healthcare Group"
    
    # LP reference
    lp_commitment_reference: str
    
    # Document metadata  
    report_quality_tier: str
    source_document_id: str
    terminology: str

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_styles():
    """Base stylesheet used across all document types."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"],
        fontSize=24, leading=30, textColor=NAVY, spaceAfter=6))
    styles.add(ParagraphStyle(
        "CoverSubtitle", parent=styles["Normal"],
        fontSize=14, leading=18, textColor=MID_GREY, spaceAfter=4))
    styles.add(ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontSize=13, leading=16, textColor=NAVY,
        spaceBefore=18, spaceAfter=8))
    styles.add(ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=7, leading=9, textColor=MID_GREY, alignment=TA_CENTER))
    styles.add(ParagraphStyle(
        "BodyJustified", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=DARK_GREY,
        alignment=TA_JUSTIFY, spaceAfter=8))
    styles.add(ParagraphStyle(
        "BodyLeft", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=DARK_GREY,
        alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(
        "SmallGrey", parent=styles["Normal"],
        fontSize=8, leading=10, textColor=MID_GREY))

    return styles

def fmt_mm(val, currency="$"):
    """Format a value in millions with comma separator."""
    if val == 0:
        return f"{currency}0.0"
    return f"{currency}{val:,.1f}"

def _get_currency_symbol(currency: str) -> str:
    return CURRENCY_SYMBOLS.get(currency, "$")

def _get_terms(terminology: str) -> dict:
    return TERMINOLOGY_SETS.get(terminology, TERMINOLOGY_SETS["A"])

def _format_date_long(date_str: str) -> str:
    """Convert '2025-11-02' to 'November 2, 2025'."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%B %d, %Y").replace(" 0", " ")

# ---------------------------------------------------------------------------
# Bank Details Generation
# ---------------------------------------------------------------------------

def generate_bank_details(fund_def: FundDefinition, rng: random.Random) -> dict:
    """Generate realistic but synthetic bank details with format variations."""
    
    # Different bank detail patterns by currency/region
    if fund_def.currency == "USD":
        banks = [
            {"name": "JPMorgan Chase Bank", "routing": "021000021"},
            {"name": "Bank of America", "routing": "026009593"},  
            {"name": "Wells Fargo Bank", "routing": "121000248"},
            {"name": "Citibank N.A.", "routing": "021000089"},
        ]
        bank = rng.choice(banks)
        
        return {
            "bank_name": bank["name"],
            "account_name": f"{fund_def.fund_name} Capital Account",
            "account_number": f"{rng.randint(100000000, 999999999)}",
            "routing_number": bank["routing"],
            "swift_code": None,
            "iban": None,
        }
        
    elif fund_def.currency == "EUR":
        banks = [
            {"name": "BNP Paribas", "swift": "BNPAFRPPXXX"},
            {"name": "Deutsche Bank AG", "swift": "DEUTDEFFXXX"},
            {"name": "ING Bank N.V.", "swift": "INGBNL2AXXX"},
        ]
        bank = rng.choice(banks)
        
        # Generate IBAN (simplified, not checksummed)
        country_code = rng.choice(["DE", "FR", "NL"])
        check_digits = f"{rng.randint(10,99)}"
        bank_code = f"{rng.randint(1000,9999)}"
        account_num = f"{rng.randint(100000000,999999999)}"
        iban = f"{country_code}{check_digits}{bank_code}{account_num}"
        
        return {
            "bank_name": bank["name"],
            "account_name": f"{fund_def.fund_name} Capital Account",
            "account_number": account_num,
            "routing_number": None,
            "swift_code": bank["swift"],
            "iban": iban,
        }
        
    elif fund_def.currency == "GBP":
        banks = [
            {"name": "Barclays Bank PLC", "swift": "BARCGB22XXX"},
            {"name": "HSBC Bank PLC", "swift": "HBUKGB4BXXX"},
            {"name": "NatWest Bank PLC", "swift": "NWBKGB2LXXX"},
        ]
        bank = rng.choice(banks)
        
        # UK sort code and account number
        sort_code = f"{rng.randint(10,99)}-{rng.randint(10,99)}-{rng.randint(10,99)}"
        
        return {
            "bank_name": bank["name"],
            "account_name": f"{fund_def.fund_name} Capital Account",
            "account_number": f"{rng.randint(10000000,99999999)}",
            "routing_number": sort_code,  # Sort code in routing field
            "swift_code": bank["swift"],
            "iban": None,
        }
    
    # Fallback for unknown currency
    return {
        "bank_name": "Generic Bank",
        "account_name": f"{fund_def.fund_name} Capital Account",
        "account_number": f"{rng.randint(100000000,999999999)}",
        "routing_number": None,
        "swift_code": None,
        "iban": None,
    }

# ---------------------------------------------------------------------------
# CAPITAL CALL NOTICES
# ---------------------------------------------------------------------------

def generate_institutional_capital_call(
    notice: CapitalCallNotice,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
):
    """Generate institutional-quality capital call notice PDF."""
    styles = get_styles()
    terms = _get_terms(notice.terminology)
    sym = _get_currency_symbol(notice.currency)
    errs = transcription_errors or {}
    story = []

    # Header
    story.append(Paragraph(notice.gp_name, styles["CoverTitle"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=12))
    
    story.append(Paragraph("CAPITAL CALL NOTICE", ParagraphStyle(
        "NoticeType", parent=styles["SectionHeading"], 
        fontSize=16, textColor=NAVY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph(notice.fund_name, ParagraphStyle(
        "FundName", parent=styles["CoverSubtitle"],
        fontSize=16, textColor=DARK_GREY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3 * inch))

    # Key details table
    call_amount_display = errs.get("call_amount_mm", notice.call_amount_mm)
    
    details_data = [
        ["Call Date", _format_date_long(notice.call_date)],
        ["Payment Due Date", _format_date_long(notice.due_date)], 
        ["Amount Called", f"{fmt_mm(call_amount_display, sym)} million"],
        ["Percentage of Commitment", f"{notice.call_amount_pct:.1f}%"],
        ["Cumulative Called to Date", f"{fmt_mm(notice.cumulative_called_mm, sym)} million"],
        ["Unfunded Commitment Remaining", f"{fmt_mm(notice.unfunded_commitment_mm, sym)} million"],
        ["LP Commitment Reference", notice.lp_commitment_reference],
    ]
    
    t = Table(details_data, colWidths=[2.5 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # Banking instructions
    story.append(Paragraph("PAYMENT INSTRUCTIONS", styles["SectionHeading"]))
    story.append(Paragraph(
        "Please transfer your capital contribution to the following account:",
        styles["BodyLeft"]))
    story.append(Spacer(1, 0.1 * inch))

    bank_data = [
        ["Bank Name", notice.bank_name],
        ["Account Name", notice.account_name],
        ["Account Number", notice.account_number],
    ]
    
    if notice.routing_number:
        if notice.currency == "GBP":
            bank_data.append(["Sort Code", notice.routing_number])
        else:
            bank_data.append(["Routing Number", notice.routing_number])
    
    if notice.swift_code:
        bank_data.append(["SWIFT Code", notice.swift_code])
        
    if notice.iban:
        bank_data.append(["IBAN", notice.iban])

    t = Table(bank_data, colWidths=[2.0 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    # Important notes
    story.append(Paragraph(
        f"<b>IMPORTANT:</b> Please ensure your wire transfer is received by "
        f"{_format_date_long(notice.due_date)}. Include your LP commitment reference "
        f"'{notice.lp_commitment_reference}' in the wire transfer details.",
        styles["BodyJustified"]))
    
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "If you have any questions regarding this capital call, please contact "
        "our investor relations team immediately.",
        styles["BodyLeft"]))

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Sincerely,", styles["BodyLeft"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>{notice.gp_name}</b>", styles["BodyLeft"]))
    
    # Footer disclaimer
    story.append(Spacer(1, 0.8 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=6))
    story.append(Paragraph(
        "This capital call notice is confidential and intended solely for "
        "the addressee. Failure to fund may result in penalties as outlined "
        "in the Limited Partnership Agreement.",
        styles["Disclaimer"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    doc.build(story)


def generate_narrative_capital_call(
    notice: CapitalCallNotice,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
):
    """Generate narrative-style capital call notice (letter format)."""
    styles = get_styles()
    sym = _get_currency_symbol(notice.currency)
    rng = random.Random(notice.source_document_id or notice.fund_id)
    story = []

    # Letterhead  
    story.append(Paragraph(f"<b>{notice.gp_name}</b>", ParagraphStyle(
        "Letterhead", parent=styles["Normal"],
        fontSize=16, leading=20, textColor=NAVY, spaceAfter=2)))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=6))

    address = GP_ADDRESSES.get(notice.gp_name,
        "300 Park Avenue, 25th Floor | New York, NY 10022 | Tel: (212) 555-0142")
    story.append(Paragraph(
        address.replace("|", "&nbsp;&nbsp;|&nbsp;&nbsp;"),
        ParagraphStyle("Addr", parent=styles["Normal"],
                       fontSize=8, leading=10, textColor=MID_GREY)))
    story.append(Spacer(1, 0.4 * inch))

    # Date and addressee
    story.append(Paragraph(_format_date_long(notice.call_date), styles["BodyLeft"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("To the Limited Partners of", styles["BodyLeft"]))
    story.append(Paragraph(f"<b>{notice.fund_name}</b>", styles["BodyLeft"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Dear Partners,", styles["BodyJustified"]))
    story.append(Spacer(1, 0.1 * inch))

    # Body paragraphs
    story.append(Paragraph(
        f"We are writing to request a capital contribution to {notice.fund_name}. "
        f"This call notice requires payment of {fmt_mm(notice.call_amount_mm, sym)} million, "
        f"representing {notice.call_amount_pct:.1f}% of your total commitment.",
        styles["BodyJustified"]))
    
    story.append(Paragraph(
        f"<b>Payment Details:</b>", styles["SectionHeading"]))
    story.append(Paragraph(
        f"Amount Due: {fmt_mm(notice.call_amount_mm, sym)} million<br/>"
        f"Payment Due Date: {_format_date_long(notice.due_date)}<br/>"
        f"Your LP Reference: {notice.lp_commitment_reference}",
        styles["BodyLeft"]))

    story.append(Paragraph(
        f"Following this call, your cumulative capital contributions will total "
        f"{fmt_mm(notice.cumulative_called_mm + notice.call_amount_mm, sym)} million, "
        f"with {fmt_mm(notice.unfunded_commitment_mm - notice.call_amount_mm, sym)} million "
        f"remaining unfunded under your commitment.",
        styles["BodyJustified"]))

    # Banking details in prose
    story.append(Paragraph("<b>Wire Transfer Instructions:</b>", styles["SectionHeading"]))
    wire_text = f"Please wire funds to {notice.bank_name}, account name '{notice.account_name}', account number {notice.account_number}"
    
    if notice.routing_number and notice.currency == "USD":
        wire_text += f", routing number {notice.routing_number}"
    elif notice.routing_number and notice.currency == "GBP":  
        wire_text += f", sort code {notice.routing_number}"
        
    if notice.swift_code:
        wire_text += f", SWIFT {notice.swift_code}"
    if notice.iban:
        wire_text += f", IBAN {notice.iban}"
        
    wire_text += f". Please reference '{notice.lp_commitment_reference}' in your wire details."
    
    story.append(Paragraph(wire_text, styles["BodyJustified"]))

    # Closing
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Please contact our investor relations team if you have any questions "
        "regarding this capital call.",
        styles["BodyJustified"]))
    
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Best regards,", styles["BodyLeft"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>{notice.gp_name}</b>", styles["BodyLeft"]))

    # Footer
    story.append(Spacer(1, 0.6 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=6))
    story.append(Paragraph(
        "This notice is confidential and intended for limited partners only.",
        styles["Disclaimer"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch)
    doc.build(story)


def generate_poor_capital_call(
    notice: CapitalCallNotice,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
):
    """Generate poor quality capital call notice (email-style, inconsistent)."""
    styles = get_styles()
    sym = _get_currency_symbol(notice.currency)
    errs = transcription_errors or {}

    plain = ParagraphStyle("Plain", parent=styles["Normal"],
                           fontSize=11, leading=14, textColor=colors.black)
    plain_small = ParagraphStyle("PlainSmall", parent=styles["Normal"],
                                 fontSize=10, leading=13, textColor=colors.black)
    bold_title = ParagraphStyle("BoldTitle", parent=styles["Normal"],
                                fontSize=14, leading=18, textColor=colors.black,
                                fontName="Helvetica-Bold")
    story = []

    # Email-style header
    story.append(Paragraph(f"From: {notice.gp_name}", plain_small))
    story.append(Paragraph(f"Re: {notice.fund_name} - Capital Call", bold_title))
    story.append(Paragraph(f"Date: {notice.call_date}", plain_small))
    story.append(Spacer(1, 0.3 * inch))

    # Casual body
    call_amount_display = errs.get("call_amount_mm", notice.call_amount_mm)
    
    story.append(Paragraph("Hi,", plain))
    story.append(Paragraph(
        f"We need {sym}{call_amount_display:.1f}mm from you for the fund. "
        f"This is {notice.call_amount_pct:.1f}% of your commitment.",
        plain))
    
    story.append(Paragraph(
        f"Due date: {notice.due_date}<br/>"
        f"Your ref: {notice.lp_commitment_reference}",
        plain_small))
    
    story.append(Paragraph(
        f"Total called so far will be {notice.cumulative_called_mm + notice.call_amount_mm:.1f}mm after this. "
        f"Still {notice.unfunded_commitment_mm - notice.call_amount_mm:.1f}mm left on your commitment.",
        plain))
        
    # Banking - inconsistent format
    story.append(Paragraph("Wire details:", plain))
    story.append(Paragraph(f"Bank: {notice.bank_name}", plain_small))
    story.append(Paragraph(f"Account: {notice.account_number} ({notice.account_name})", plain_small))
    
    if notice.routing_number:
        if notice.currency == "USD":
            story.append(Paragraph(f"Routing: {notice.routing_number}", plain_small))
        elif notice.currency == "GBP":
            story.append(Paragraph(f"Sort code: {notice.routing_number}", plain_small))
            
    if notice.swift_code:
        story.append(Paragraph(f"SWIFT: {notice.swift_code}", plain_small))
    if notice.iban:
        story.append(Paragraph(f"IBAN: {notice.iban}", plain_small))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Questions? Just reply.", plain))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Thanks,", plain))
    story.append(Paragraph(notice.gp_name.split()[0], plain))  # Just first word of GP name

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=1.0 * inch, bottomMargin=1.0 * inch,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch)
    doc.build(story)

# ---------------------------------------------------------------------------
# DISTRIBUTION NOTICES  
# ---------------------------------------------------------------------------

def generate_institutional_distribution(
    notice: DistributionNotice,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
):
    """Generate institutional-quality distribution notice PDF."""
    styles = get_styles()
    terms = _get_terms(notice.terminology)
    sym = _get_currency_symbol(notice.currency)
    errs = transcription_errors or {}
    story = []

    # Header
    story.append(Paragraph(notice.gp_name, styles["CoverTitle"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=12))
    
    story.append(Paragraph("DISTRIBUTION NOTICE", ParagraphStyle(
        "NoticeType", parent=styles["SectionHeading"], 
        fontSize=16, textColor=NAVY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph(notice.fund_name, ParagraphStyle(
        "FundName", parent=styles["CoverSubtitle"],
        fontSize=16, textColor=DARK_GREY, alignment=TA_CENTER)))
    story.append(Spacer(1, 0.3 * inch))

    # Distribution type mapping for display
    type_display = {
        "return_of_capital": "Return of Capital",
        "income": "Income Distribution", 
        "gain": "Capital Gain Distribution"
    }
    
    # Key details table
    dist_amount_display = errs.get("distribution_amount_mm", notice.distribution_amount_mm)
    dist_type_display = errs.get("distribution_type", notice.distribution_type)
    
    details_data = [
        ["Distribution Date", _format_date_long(notice.distribution_date)],
        ["Distribution Amount", f"{fmt_mm(dist_amount_display, sym)} million"],
        ["Distribution Type", type_display.get(dist_type_display, dist_type_display)],
        ["Cumulative Distributions to Date", f"{fmt_mm(notice.cumulative_distributed_mm, sym)} million"],
        ["LP Commitment Reference", notice.lp_commitment_reference],
    ]
    
    if notice.realization_source:
        details_data.insert(-1, ["Source of Distribution", notice.realization_source])
    
    t = Table(details_data, colWidths=[2.8 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # Tax implications section
    story.append(Paragraph("TAX CONSIDERATIONS", styles["SectionHeading"]))
    if notice.distribution_type == "return_of_capital":
        tax_note = (
            "This distribution represents a return of your original capital contribution "
            "and is generally not taxable. It reduces your cost basis in the fund."
        )
    elif notice.distribution_type == "income":
        tax_note = (
            "This distribution represents income generated by the fund and may be "
            "subject to taxation as ordinary income. Please consult your tax advisor."
        )
    else:  # gain
        tax_note = (
            "This distribution represents capital gains from fund investments and may "
            "be subject to capital gains taxation. Please consult your tax advisor."
        )
    
    story.append(Paragraph(tax_note, styles["BodyJustified"]))
    story.append(Spacer(1, 0.2 * inch))

    # Wire transfer info
    story.append(Paragraph(
        f"The distribution will be wired to your account on record on "
        f"{_format_date_long(notice.distribution_date)}. Please ensure your "
        f"banking information is current with our transfer agent.",
        styles["BodyJustified"]))
    
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "If you have any questions regarding this distribution, please contact "
        "our investor relations team.",
        styles["BodyLeft"]))

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Sincerely,", styles["BodyLeft"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>{notice.gp_name}</b>", styles["BodyLeft"]))
    
    # Footer disclaimer
    story.append(Spacer(1, 0.8 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=6))
    story.append(Paragraph(
        "This distribution notice is confidential and intended solely for "
        "the addressee. Tax treatment may vary based on individual circumstances. "
        "Please consult your tax advisor.",
        styles["Disclaimer"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    doc.build(story)


def generate_narrative_distribution(
    notice: DistributionNotice,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
):
    """Generate narrative-style distribution notice (letter format)."""
    styles = get_styles()
    sym = _get_currency_symbol(notice.currency)
    story = []

    # Letterhead  
    story.append(Paragraph(f"<b>{notice.gp_name}</b>", ParagraphStyle(
        "Letterhead", parent=styles["Normal"],
        fontSize=16, leading=20, textColor=NAVY, spaceAfter=2)))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=6))

    address = GP_ADDRESSES.get(notice.gp_name,
        "300 Park Avenue, 25th Floor | New York, NY 10022 | Tel: (212) 555-0142")
    story.append(Paragraph(
        address.replace("|", "&nbsp;&nbsp;|&nbsp;&nbsp;"),
        ParagraphStyle("Addr", parent=styles["Normal"],
                       fontSize=8, leading=10, textColor=MID_GREY)))
    story.append(Spacer(1, 0.4 * inch))

    # Date and addressee
    story.append(Paragraph(_format_date_long(notice.distribution_date), styles["BodyLeft"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("To the Limited Partners of", styles["BodyLeft"]))
    story.append(Paragraph(f"<b>{notice.fund_name}</b>", styles["BodyLeft"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Dear Partners,", styles["BodyJustified"]))
    story.append(Spacer(1, 0.1 * inch))

    # Body paragraphs  
    type_prose = {
        "return_of_capital": "return of capital",
        "income": "income distribution",
        "gain": "capital gain distribution" 
    }
    
    story.append(Paragraph(
        f"We are pleased to announce a distribution from {notice.fund_name}. "
        f"This {type_prose.get(notice.distribution_type, 'distribution')} totals "
        f"{fmt_mm(notice.distribution_amount_mm, sym)} million and will be "
        f"transferred to your account on {_format_date_long(notice.distribution_date)}.",
        styles["BodyJustified"]))
    
    if notice.realization_source:
        story.append(Paragraph(
            f"This distribution results from {notice.realization_source}, "
            f"demonstrating the continued progress of our investment strategy.",
            styles["BodyJustified"]))
    
    story.append(Paragraph(
        f"Following this distribution, cumulative distributions to limited partners "
        f"will total {fmt_mm(notice.cumulative_distributed_mm + notice.distribution_amount_mm, sym)} million "
        f"since the fund's inception.",
        styles["BodyJustified"]))

    # Tax paragraph
    story.append(Paragraph("<b>Tax Implications:</b>", styles["SectionHeading"]))
    if notice.distribution_type == "return_of_capital":
        tax_text = (
            "This distribution represents a return of your original capital and "
            "typically reduces your cost basis in the fund without immediate tax consequences."
        )
    elif notice.distribution_type == "income":
        tax_text = (
            "This income distribution may be subject to ordinary income tax rates. "
            "You will receive appropriate tax documentation for your filings."
        )
    else:
        tax_text = (
            "This capital gain distribution may be subject to capital gains tax treatment. "
            "Please review the tax documentation that will be provided."
        )
    
    story.append(Paragraph(tax_text, styles["BodyJustified"]))

    # Closing
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "We appreciate your continued partnership and commitment to the fund. "
        "Please contact our investor relations team with any questions.",
        styles["BodyJustified"]))
    
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Sincerely,", styles["BodyLeft"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"<b>{notice.gp_name}</b>", styles["BodyLeft"]))

    # Footer
    story.append(Spacer(1, 0.6 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=6))
    story.append(Paragraph(
        "This notice is confidential and intended for limited partners only. "
        "Please consult your tax advisor regarding the treatment of this distribution.",
        styles["Disclaimer"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch)
    doc.build(story)


def generate_poor_distribution(
    notice: DistributionNotice,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
):
    """Generate poor quality distribution notice (email-style, inconsistent)."""
    styles = get_styles()
    sym = _get_currency_symbol(notice.currency)
    errs = transcription_errors or {}

    plain = ParagraphStyle("Plain", parent=styles["Normal"],
                           fontSize=11, leading=14, textColor=colors.black)
    plain_small = ParagraphStyle("PlainSmall", parent=styles["Normal"],
                                 fontSize=10, leading=13, textColor=colors.black)
    bold_title = ParagraphStyle("BoldTitle", parent=styles["Normal"],
                                fontSize=14, leading=18, textColor=colors.black,
                                fontName="Helvetica-Bold")
    story = []

    # Email-style header
    story.append(Paragraph(f"From: {notice.gp_name}", plain_small))
    story.append(Paragraph(f"Re: {notice.fund_name} - Distribution", bold_title))
    story.append(Paragraph(f"Date: {notice.distribution_date}", plain_small))
    story.append(Spacer(1, 0.3 * inch))

    # Casual body
    dist_amount_display = errs.get("distribution_amount_mm", notice.distribution_amount_mm)
    
    story.append(Paragraph("Hi,", plain))
    story.append(Paragraph(
        f"Good news - we're sending you {sym}{dist_amount_display:.1f}mm from the fund. "
        f"Will hit your account on {notice.distribution_date}.",
        plain))
    
    # Distribution type - often unclear in poor quality docs
    if notice.distribution_type == "return_of_capital":
        type_text = "This is return of capital (not taxable usually)"
    elif notice.distribution_type == "income":
        type_text = "This is income - might be taxable"
    else:
        type_text = "Capital gains distribution"
        
    story.append(Paragraph(type_text, plain_small))
    
    if notice.realization_source:
        story.append(Paragraph(f"From: {notice.realization_source}", plain_small))
    
    story.append(Paragraph(
        f"Total distributions now: {notice.cumulative_distributed_mm + notice.distribution_amount_mm:.1f}mm<br/>"
        f"Your ref: {notice.lp_commitment_reference}",
        plain_small))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Check with your tax guy about tax stuff.", plain_small))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Let me know if questions.", plain))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Best,", plain))
    story.append(Paragraph(notice.gp_name.split()[0], plain))  # Just first word

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=1.0 * inch, bottomMargin=1.0 * inch,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch)
    doc.build(story)

# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------

CAPITAL_CALL_GENERATORS = {
    "institutional": generate_institutional_capital_call,
    "narrative": generate_narrative_capital_call,
    "poor": generate_poor_capital_call,
}

DISTRIBUTION_GENERATORS = {
    "institutional": generate_institutional_distribution,
    "narrative": generate_narrative_distribution,
    "poor": generate_poor_distribution,
}