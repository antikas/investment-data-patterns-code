"""
Three PDF generators for GP quarterly reports, one per quality tier.

Preserves all reportlab styling from the original scripts/generate_pdfs.py.
Extended with:
- Terminology dicts (field labels from fund's terminology set)
- Currency symbols (USD/EUR/GBP)
- QuarterSnapshot input (not FundGroundTruth)
- Portfolio schedule control (institutional only)
- Narrative variation
- Transcription error injection (PDF shows wrong number; ground truth correct)
- IRR ambiguity injection
- Report date in narrative letters
"""

import random
from typing import Dict, Optional

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
from shared.simulation import QuarterSnapshot


# ---------------------------------------------------------------------------
# Colour palette (from original)
# ---------------------------------------------------------------------------
NAVY = HexColor("#1B2A4A")
DARK_GREY = HexColor("#333333")
MID_GREY = HexColor("#666666")
TABLE_HEADER_BG = HexColor("#1B2A4A")
TABLE_ALT_ROW = HexColor("#F0F4F8")


# ---------------------------------------------------------------------------
# Shared styles and formatters
# ---------------------------------------------------------------------------

def get_styles():
    """Base stylesheet used across all tiers."""
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


def fmt_pct(val):
    return f"{val:.1f}%"


def fmt_multiple(val):
    return f"{val:.2f}x"


def _get_currency_symbol(currency: str) -> str:
    return CURRENCY_SYMBOLS.get(currency, "$")


def _get_terms(fund_def: FundDefinition) -> dict:
    return TERMINOLOGY_SETS.get(fund_def.terminology, TERMINOLOGY_SETS["A"])


def _format_report_date_long(report_date: str) -> str:
    """Convert '2025-11-02' to 'November 2, 2025'."""
    from datetime import datetime
    dt = datetime.strptime(report_date, "%Y-%m-%d")
    return dt.strftime("%B %d, %Y").replace(" 0", " ")


# ---------------------------------------------------------------------------
# INSTITUTIONAL TIER
# ---------------------------------------------------------------------------

def _inst_header_footer(canvas, doc, snapshot, terms):
    """Header/footer on each page for institutional reports."""
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(1.5)
    canvas.line(54, letter[1] - 50, letter[0] - 54, letter[1] - 50)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(54, letter[1] - 45, snapshot.gp_name)
    canvas.drawRightString(
        letter[0] - 54, letter[1] - 45,
        f"{snapshot.fund_name} | {snapshot.reporting_period}")
    canvas.setLineWidth(0.5)
    canvas.line(54, 45, letter[0] - 54, 45)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(54, 33, "CONFIDENTIAL \u2014 For Limited Partner Use Only")
    canvas.drawRightString(letter[0] - 54, 33, f"Page {doc.page}")
    canvas.restoreState()


def generate_institutional_pdf(
    snapshot: QuarterSnapshot,
    fund_def: FundDefinition,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
    irr_ambiguous: bool = False,
):
    """Generate an ILPA-standard quarterly report PDF."""
    styles = get_styles()
    terms = _get_terms(fund_def)
    sym = _get_currency_symbol(snapshot.currency)
    errs = transcription_errors or {}
    story = []

    # --- Cover page ---
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(snapshot.gp_name, styles["CoverTitle"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="60%", thickness=2, color=NAVY, spaceAfter=12))
    story.append(Paragraph(snapshot.fund_name, ParagraphStyle(
        "FundNameCover", parent=styles["CoverSubtitle"],
        fontSize=18, leading=22, textColor=DARK_GREY)))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Quarterly Report", ParagraphStyle(
        "QRLabel", parent=styles["CoverSubtitle"],
        fontSize=14, textColor=MID_GREY)))
    story.append(Paragraph(
        f"For the Period Ending {snapshot.quarter_end_date}",
        styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Strategy: {snapshot.strategy}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
        f"Vintage: {snapshot.vintage_year}&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
        f"Currency: {snapshot.currency}",
        ParagraphStyle("CoverMeta", parent=styles["SmallGrey"],
                       fontSize=10, textColor=MID_GREY)))
    story.append(Spacer(1, 2.5 * inch))
    story.append(Paragraph(
        "This report is confidential and intended solely for the use of "
        "limited partners. Past performance is not indicative of future results.",
        styles["Disclaimer"]))
    story.append(PageBreak())

    # --- Fund Summary ---
    story.append(Paragraph(terms["fund_summary_title"], styles["SectionHeading"]))
    summary_data = [
        ["Metric", "Value"],
        [terms["committed"], fmt_mm(snapshot.committed_capital_mm, sym) + "mm"],
        [terms["called"], fmt_mm(snapshot.called_capital_mm, sym) + "mm"],
        [terms["distributed"], fmt_mm(snapshot.distributed_capital_mm, sym) + "mm"],
        [terms["nav"], fmt_mm(snapshot.nav_mm, sym) + "mm"],
        ["Unfunded Commitment",
         fmt_mm(snapshot.committed_capital_mm - snapshot.called_capital_mm, sym) + "mm"],
    ]
    t = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BACKGROUND", (0, 1), (-1, 1), TABLE_ALT_ROW),
        ("BACKGROUND", (0, 3), (-1, 3), TABLE_ALT_ROW),
        ("BACKGROUND", (0, 5), (-1, 5), TABLE_ALT_ROW),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # --- Performance metrics ---
    story.append(Paragraph(terms["performance_title"], styles["SectionHeading"]))

    display_tvpi = errs.get("tvpi", snapshot.tvpi)
    perf_data = [
        [terms["irr_net"], terms["tvpi"], terms["dpi"], terms["rvpi"]],
        [fmt_pct(snapshot.net_irr_pct), fmt_multiple(display_tvpi),
         fmt_multiple(snapshot.dpi), fmt_multiple(snapshot.rvpi)],
    ]
    t = Table(perf_data, colWidths=[1.5 * inch] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Note: Performance metrics are calculated net of management fees, "
        "fund expenses, and carried interest. IRR is computed using daily "
        "cash flows since inception.",
        styles["SmallGrey"]))
    story.append(Spacer(1, 0.3 * inch))

    # --- Schedule of Investments (optional) ---
    if fund_def.include_portfolio_schedule and snapshot.portfolio_companies:
        story.append(Paragraph(terms["schedule_title"], styles["SectionHeading"]))

        inv_header = ["Company", "Sector", "Investment\nDate",
                      f"Cost\n({sym}mm)", f"Fair Value\n({sym}mm)", "% of\nNAV"]
        inv_rows = [inv_header]
        total_cost, total_fv = 0.0, 0.0
        for pc in snapshot.portfolio_companies:
            # Check for transcription error on this company
            err_key = f"pc_fair_value_{pc.name}"
            display_fv = errs.get(err_key, pc.fair_value_mm)

            pct_nav = display_fv / snapshot.nav_mm * 100 if snapshot.nav_mm > 0 else 0
            inv_rows.append([
                pc.name, pc.sector, pc.investment_date,
                f"{sym}{pc.initial_cost_mm:,.1f}", f"{sym}{display_fv:,.1f}",
                f"{pct_nav:.1f}%"])
            total_cost += pc.initial_cost_mm
            total_fv += display_fv

        inv_rows.append([
            "Total Portfolio", "", "",
            f"{sym}{total_cost:,.1f}", f"{sym}{total_fv:,.1f}",
            f"{total_fv / snapshot.nav_mm * 100:.1f}%" if snapshot.nav_mm > 0 else "0.0%"])

        col_widths = [2.0 * inch, 1.4 * inch, 0.9 * inch,
                      0.85 * inch, 0.85 * inch, 0.6 * inch]
        t = Table(inv_rows, colWidths=col_widths)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 1, NAVY),
        ]
        for i in range(1, len(inv_rows) - 1):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_ROW))
        t.setStyle(TableStyle(style_cmds))
        story.append(t)

    story.append(PageBreak())

    # --- Fee & Expense Summary ---
    story.append(Paragraph(terms["fees_title"], styles["SectionHeading"]))
    story.append(Paragraph(
        f"For the period ending {snapshot.quarter_end_date}", styles["SmallGrey"]))
    story.append(Spacer(1, 0.1 * inch))

    total_fees = snapshot.management_fee_mm + snapshot.carried_interest_mm + snapshot.other_expenses_mm
    committed = snapshot.committed_capital_mm
    fee_data = [
        ["Item", f"Amount ({sym}mm)", "% of Committed"],
        [terms["management_fee"], f"{sym}{snapshot.management_fee_mm:,.2f}",
         f"{snapshot.management_fee_mm / committed * 100:.2f}%"],
        [terms["carried_interest"], f"{sym}{snapshot.carried_interest_mm:,.2f}",
         f"{snapshot.carried_interest_mm / committed * 100:.2f}%"],
        ["Other Fund Expenses", f"{sym}{snapshot.other_expenses_mm:,.2f}",
         f"{snapshot.other_expenses_mm / committed * 100:.2f}%"],
        ["Total", f"{sym}{total_fees:,.2f}",
         f"{total_fees / committed * 100:.2f}%"],
    ]
    t = Table(fee_data, colWidths=[2.5 * inch, 1.8 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 2), (-1, 2), TABLE_ALT_ROW),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("LINEABOVE", (0, -1), (-1, -1), 1, NAVY),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # --- Capital Account Statement ---
    story.append(Paragraph(terms["capital_account_title"], styles["SectionHeading"]))
    story.append(Paragraph(
        f"Illustrative LP commitment of {fmt_mm(snapshot.lp_commitment_mm, sym)}mm "
        f"(of {fmt_mm(snapshot.committed_capital_mm, sym)}mm total fund)",
        styles["SmallGrey"]))
    story.append(Spacer(1, 0.1 * inch))

    lp_ratio = snapshot.lp_commitment_mm / snapshot.committed_capital_mm
    pcap_data = [
        ["", f"Amount ({sym}mm)"],
        ["Commitment", f"{sym}{snapshot.lp_commitment_mm:,.2f}"],
        ["Cumulative Capital Called", f"{sym}{snapshot.called_capital_mm * lp_ratio:,.2f}"],
        ["Cumulative Distributions", f"({sym}{snapshot.distributed_capital_mm * lp_ratio:,.2f})"],
        [terms["nav"], f"{sym}{snapshot.nav_mm * lp_ratio:,.2f}"],
        ["Remaining Unfunded",
         f"{sym}{snapshot.lp_commitment_mm - snapshot.called_capital_mm * lp_ratio:,.2f}"],
    ]
    t = Table(pcap_data, colWidths=[3.0 * inch, 2.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, 1), TABLE_ALT_ROW),
        ("BACKGROUND", (0, 3), (-1, 3), TABLE_ALT_ROW),
        ("BACKGROUND", (0, 5), (-1, 5), TABLE_ALT_ROW),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "This report has been prepared by the General Partner on an unaudited "
        "basis. Valuations reflect the General Partner's best estimate of fair "
        "value as of the reporting date and are subject to change.",
        styles["SmallGrey"]))

    # Build
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.85 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch)

    def on_page(canvas, doc_obj):
        _inst_header_footer(canvas, doc_obj, snapshot, terms)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


# ---------------------------------------------------------------------------
# NARRATIVE TIER
# ---------------------------------------------------------------------------

def generate_narrative_pdf(
    snapshot: QuarterSnapshot,
    fund_def: FundDefinition,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
    irr_ambiguous: bool = False,
):
    """Generate a GP quarterly letter PDF with numbers embedded in prose."""
    styles = get_styles()
    terms = _get_terms(fund_def)
    sym = _get_currency_symbol(snapshot.currency)
    rng = random.Random(snapshot.source_document_id or snapshot.fund_id)
    story = []

    # Letterhead
    story.append(Paragraph(
        f"<b>{snapshot.gp_name}</b>",
        ParagraphStyle("Letterhead", parent=styles["Normal"],
                       fontSize=16, leading=20, textColor=NAVY, spaceAfter=2)))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=6))

    address = GP_ADDRESSES.get(snapshot.gp_name,
        "300 Park Avenue, 25th Floor | New York, NY 10022 | Tel: (212) 555-0142")
    story.append(Paragraph(
        address.replace("|", "&nbsp;&nbsp;|&nbsp;&nbsp;"),
        ParagraphStyle("Addr", parent=styles["Normal"],
                       fontSize=8, leading=10, textColor=MID_GREY)))
    story.append(Spacer(1, 0.4 * inch))

    # Date and addressee — use actual report_date
    letter_date = _format_report_date_long(snapshot.report_date) if snapshot.report_date else "Date not available"
    story.append(Paragraph(letter_date, styles["BodyLeft"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("To the Limited Partners of", styles["BodyLeft"]))
    story.append(Paragraph(f"<b>{snapshot.fund_name}</b>", styles["BodyLeft"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Dear Partners,", styles["BodyJustified"]))
    story.append(Spacer(1, 0.1 * inch))

    # Vary opening: some lead with market commentary, some with fund numbers
    lead_with_market = rng.random() > 0.5

    if lead_with_market:
        market_openers = [
            f"Market conditions during {snapshot.reporting_period} presented both opportunities "
            f"and challenges across the {snapshot.strategy.lower()} landscape. Against this backdrop, "
            f"we are pleased to share the quarterly update for {snapshot.fund_name}.",
            f"The quarter ending {snapshot.quarter_end_date} was marked by continued evolution "
            f"in our target markets. We write to update you on the progress of {snapshot.fund_name} "
            f"during this period.",
        ]
        story.append(Paragraph(rng.choice(market_openers), styles["BodyJustified"]))

    drawdown_pct = snapshot.called_capital_mm / snapshot.committed_capital_mm * 100

    story.append(Paragraph(
        f"<b>Fund Overview</b>", styles["SectionHeading"]))
    story.append(Paragraph(
        f"As of quarter end, the fund has called {fmt_mm(snapshot.called_capital_mm, sym)} million "
        f"of the {fmt_mm(snapshot.committed_capital_mm, sym)} million in {terms['committed'].lower()}, "
        f"representing a {drawdown_pct:.0f}% draw-down ratio. We have returned "
        f"{fmt_mm(snapshot.distributed_capital_mm, sym)} million to partners to date. "
        f"The fund's {terms['nav'].lower()} stands at {fmt_mm(snapshot.nav_mm, sym)} million "
        f"as of {snapshot.quarter_end_date}.",
        styles["BodyJustified"]))

    # IRR paragraph — may be ambiguous
    if irr_ambiguous:
        # Show both IRR values without labelling gross vs net
        story.append(Paragraph(
            f"The fund has generated returns of {snapshot.gross_irr_pct:.1f}% / "
            f"{snapshot.net_irr_pct:.1f}% since inception, with a total value "
            f"multiple of {snapshot.tvpi:.2f}x.",
            styles["BodyJustified"]))
    else:
        story.append(Paragraph(
            f"On a net basis, the fund has generated an {terms['irr_net'].lower()} of "
            f"{snapshot.net_irr_pct:.1f}% since inception, with a {terms['tvpi'].lower()} of "
            f"{snapshot.tvpi:.2f}x. Cash-on-cash returns ({terms['dpi']}) stand at "
            f"{snapshot.dpi:.2f}x, with residual value ({terms['rvpi']}) of {snapshot.rvpi:.2f}x.",
            styles["BodyJustified"]))

    # Portfolio commentary
    if snapshot.portfolio_companies:
        story.append(Paragraph("<b>Portfolio Update</b>", styles["SectionHeading"]))
        for pc in snapshot.portfolio_companies:
            gain_loss = pc.fair_value_mm - pc.initial_cost_mm
            gain_loss_pct = gain_loss / pc.initial_cost_mm * 100 if pc.initial_cost_mm > 0 else 0

            if gain_loss > 0:
                direction = "appreciated"
                commentary = (
                    f"The position has performed well, reflecting strong "
                    f"operational execution and favorable market conditions "
                    f"within the {pc.sector.lower()} sector.")
            elif gain_loss > -pc.initial_cost_mm * 0.1:
                direction = "remained broadly flat"
                commentary = (
                    f"Management continues to execute on the value creation "
                    f"plan. We expect performance to improve as operational "
                    f"initiatives take effect over the coming quarters.")
            else:
                direction = "declined in value"
                commentary = (
                    f"The investment has faced headwinds, including market "
                    f"softness in the {pc.sector.lower()} sector. We are "
                    f"actively working with management to address operational "
                    f"challenges and protect downside.")

            story.append(Paragraph(
                f"<b>{pc.name}</b> ({pc.sector})", styles["BodyLeft"]))
            story.append(Paragraph(
                f"{pc.name}, in which the fund invested "
                f"{fmt_mm(pc.initial_cost_mm, sym)} million in "
                f"{pc.investment_date[:4]}, has {direction} to a current "
                f"fair value of {fmt_mm(pc.fair_value_mm, sym)} million "
                f"({gain_loss_pct:+.1f}% since investment). {commentary}",
                styles["BodyJustified"]))
            story.append(Spacer(1, 0.05 * inch))

    # Fees
    story.append(Paragraph("<b>Fees and Expenses</b>", styles["SectionHeading"]))
    total_fees = snapshot.management_fee_mm + snapshot.other_expenses_mm
    carry_text = (
        f" Accrued {terms['carried_interest'].lower()} for the period was "
        f"{fmt_mm(snapshot.carried_interest_mm, sym)} million."
        if snapshot.carried_interest_mm > 0
        else f" No {terms['carried_interest'].lower()} has been accrued to date."
    )
    story.append(Paragraph(
        f"{terms['management_fee']} for the period totalled "
        f"{fmt_mm(snapshot.management_fee_mm, sym)} million. Other fund-related "
        f"expenses, including legal, audit, and administration costs, were "
        f"{fmt_mm(snapshot.other_expenses_mm, sym)} million, bringing total fees "
        f"and expenses to {fmt_mm(total_fees, sym)} million for the quarter."
        f"{carry_text}",
        styles["BodyJustified"]))

    # Closing
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "We remain committed to transparent reporting and welcome any "
        "questions from our limited partners. Please do not hesitate to "
        "contact our investor relations team.",
        styles["BodyJustified"]))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Respectfully,", styles["BodyLeft"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"<b>{snapshot.gp_name}</b>", styles["BodyLeft"]))

    story.append(Spacer(1, 0.6 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceAfter=6))
    story.append(Paragraph(
        f"This letter is confidential and intended solely for the limited "
        f"partners of {snapshot.fund_name}. The information contained herein "
        f"is unaudited and subject to revision. Past performance is not "
        f"indicative of future results.",
        styles["Disclaimer"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch)
    doc.build(story)


# ---------------------------------------------------------------------------
# POOR QUALITY TIER
# ---------------------------------------------------------------------------

def generate_poor_quality_pdf(
    snapshot: QuarterSnapshot,
    fund_def: FundDefinition,
    output_path: str,
    transcription_errors: Optional[Dict[str, float]] = None,
    irr_ambiguous: bool = False,
):
    """Generate a bare-bones PDF. Inconsistent formatting, sparse structure."""
    styles = get_styles()
    terms = _get_terms(fund_def)
    sym = _get_currency_symbol(snapshot.currency)
    errs = transcription_errors or {}

    plain = ParagraphStyle("Plain", parent=styles["Normal"],
                           fontSize=11, leading=14, textColor=colors.black)
    plain_small = ParagraphStyle("PlainSmall", parent=styles["Normal"],
                                 fontSize=10, leading=13, textColor=colors.black)
    bold_title = ParagraphStyle("BoldTitle", parent=styles["Normal"],
                                fontSize=14, leading=18, textColor=colors.black,
                                fontName="Helvetica-Bold")
    story = []

    # No cover page, just starts
    story.append(Paragraph(snapshot.fund_name, bold_title))
    story.append(Paragraph(snapshot.gp_name, plain))
    story.append(Paragraph(f"Update for {snapshot.reporting_period}", plain_small))
    story.append(Spacer(1, 0.3 * inch))

    # Deliberately inconsistent formatting
    story.append(Paragraph(
        f"Fund size: {sym}{snapshot.committed_capital_mm:.0f}mm", plain))
    story.append(Paragraph(
        f"Called to date: {snapshot.called_capital_mm}", plain))
    story.append(Paragraph(
        f"{terms['nav']}: {snapshot.nav_mm}mm", plain))
    if snapshot.distributed_capital_mm > 0:
        story.append(Paragraph(
            f"Distributions: {sym}{snapshot.distributed_capital_mm}mm", plain))
    story.append(Spacer(1, 0.15 * inch))

    # Returns — use transcription error for TVPI if present
    display_tvpi = errs.get("tvpi", snapshot.tvpi)
    # IRR always shown ambiguously for poor tier (just "IRR" without gross/net)
    story.append(Paragraph(
        f"Returns: {snapshot.net_irr_pct}% IRR, {display_tvpi}x multiple", plain))
    story.append(Spacer(1, 0.15 * inch))

    # Fees
    story.append(Paragraph(
        f"Fees: {snapshot.management_fee_mm}mm mgmt fee + "
        f"{snapshot.other_expenses_mm}mm other", plain))
    story.append(Spacer(1, 0.2 * inch))

    # Portfolio companies — inconsistent formatting
    if snapshot.portfolio_companies:
        story.append(Paragraph("<b>Investments:</b>", plain))
        story.append(Spacer(1, 0.05 * inch))
        for i, pc in enumerate(snapshot.portfolio_companies):
            err_key = f"pc_fair_value_{pc.name}"
            display_fv = errs.get(err_key, pc.fair_value_mm)

            if i == 0:
                story.append(Paragraph(
                    f"&bull; {pc.name} ({pc.sector}): cost {pc.initial_cost_mm}m, "
                    f"now worth approx {display_fv}m", plain_small))
            elif i == 1:
                story.append(Paragraph(
                    f"&bull; {pc.name} - {sym}{display_fv:.0f}mm fair value "
                    f"(invested {sym}{pc.initial_cost_mm:.0f}mm)", plain_small))
            else:
                story.append(Paragraph(
                    f"&bull; {pc.name}: {display_fv}m", plain_small))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Pls reach out with any questions.", plain_small))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Best,", plain_small))
    story.append(Paragraph(snapshot.gp_name.split("(")[0].strip(), plain_small))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        topMargin=1.0 * inch, bottomMargin=1.0 * inch,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch)
    doc.build(story)


# ---------------------------------------------------------------------------
# Generator dispatch
# ---------------------------------------------------------------------------

GENERATORS = {
    "institutional": generate_institutional_pdf,
    "narrative": generate_narrative_pdf,
    "poor": generate_poor_quality_pdf,
}
