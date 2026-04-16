"""Generate PDF report from PROJECT_REPORT.md using reportlab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Preformatted, KeepTogether, HRFlowable
)
from reportlab.lib import colors
import re


WIDTH, HEIGHT = A4

# Colors
GREEN = HexColor("#2d8a4e")
DARK = HexColor("#1e1e1e")
GRAY = HexColor("#555555")
LIGHT_GRAY = HexColor("#f5f5f0")
MED_GRAY = HexColor("#999999")
TABLE_HEADER_BG = HexColor("#2d8a4e")
TABLE_ALT_BG = HexColor("#f8f8f5")


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=30,
        textColor=DARK,
        alignment=TA_CENTER,
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        fontName="Helvetica",
        fontSize=15,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CoverMeta",
        fontName="Helvetica",
        fontSize=12,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="H1",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=DARK,
        spaceBefore=16,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="H2",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=HexColor("#333333"),
        spaceBefore=12,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="H3",
        fontName="Helvetica-Bold",
        fontSize=11.5,
        textColor=HexColor("#444444"),
        spaceBefore=10,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        fontName="Helvetica",
        fontSize=10,
        textColor=HexColor("#333333"),
        spaceBefore=2,
        spaceAfter=4,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        name="BulletItem",
        fontName="Helvetica",
        fontSize=10,
        textColor=HexColor("#333333"),
        spaceBefore=1,
        spaceAfter=2,
        leading=14,
        leftIndent=16,
        bulletIndent=6,
    ))
    styles.add(ParagraphStyle(
        name="CodeStyle",
        fontName="Courier",
        fontSize=8.5,
        textColor=HexColor("#333333"),
        backColor=LIGHT_GRAY,
        spaceBefore=4,
        spaceAfter=6,
        leading=11,
        leftIndent=8,
        rightIndent=8,
    ))
    styles.add(ParagraphStyle(
        name="FooterStyle",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=MED_GRAY,
        alignment=TA_CENTER,
    ))
    return styles


def clean_md(text):
    """Convert markdown inline formatting to reportlab XML tags."""
    # Bold
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="9">\1</font>', text)
    # Links - just show text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Escape XML special chars (but not our tags)
    text = text.replace("&", "&amp;")
    # Fix double-escaped
    text = text.replace("&amp;amp;", "&amp;")
    return text


def parse_md_table(lines):
    """Parse markdown table into headers and rows."""
    headers = []
    rows = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if i == 0:
            headers = cells
        elif i == 1:
            continue
        else:
            rows.append(cells)
    return headers, rows


def build_table(headers, rows, styles):
    """Build a reportlab Table from parsed markdown table."""
    # Clean markdown from cells
    clean_headers = [re.sub(r'\*\*([^*]+)\*\*', r'\1', h) for h in headers]

    data = [clean_headers]
    for row in rows:
        clean_row = [re.sub(r'\*\*([^*]+)\*\*', r'\1', c) for c in row]
        # Pad if needed
        while len(clean_row) < len(clean_headers):
            clean_row.append("")
        data.append(clean_row[:len(clean_headers)])

    num_cols = len(clean_headers)
    avail_width = 170 * mm
    col_width = avail_width / num_cols

    # Estimate column widths based on content
    col_widths = []
    for ci in range(num_cols):
        max_len = len(clean_headers[ci])
        for row in data[1:]:
            if ci < len(row):
                max_len = max(max_len, len(row[ci]))
        col_widths.append(max_len)
    total = sum(col_widths) if sum(col_widths) > 0 else 1
    col_widths = [max(18 * mm, w / total * avail_width) for w in col_widths]
    # Re-normalize
    s = sum(col_widths)
    col_widths = [w / s * avail_width for w in col_widths]

    # Wrap cells in Paragraphs for word-wrapping
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5,
                                 textColor=HexColor("#333333"), leading=11)
    header_style = ParagraphStyle("hcell", fontName="Helvetica-Bold", fontSize=8.5,
                                   textColor=white, leading=11)

    wrapped_data = []
    for ri, row in enumerate(data):
        wrapped_row = []
        for ci, cell in enumerate(row):
            if ri == 0:
                wrapped_row.append(Paragraph(cell, header_style))
            else:
                wrapped_row.append(Paragraph(cell, cell_style))
        wrapped_data.append(wrapped_row)

    t = Table(wrapped_data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    # Alternating row colors
    for i in range(1, len(wrapped_data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_BG))

    t.setStyle(TableStyle(style_cmds))
    return t


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(MED_GRAY)
    canvas.drawCentredString(WIDTH / 2, 12 * mm, f"Page {doc.page}")
    if doc.page > 1:
        canvas.drawRightString(WIDTH - 20 * mm, HEIGHT - 12 * mm,
                               "PharmaCast - Project Report")
    canvas.restoreState()


def generate_pdf():
    doc = SimpleDocTemplate(
        "PROJECT_REPORT.pdf",
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = get_styles()
    story = []

    # ─── Cover Page ───
    story.append(Spacer(1, 60 * mm))
    story.append(Paragraph("PharmaCast", styles["CoverTitle"]))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Pharmaceutical Demand Forecasting", styles["CoverSubtitle"]))
    story.append(Paragraph("Decision Support System", styles["CoverSubtitle"]))
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="30%", thickness=1.5, color=GREEN,
                             spaceAfter=10 * mm, spaceBefore=5 * mm))
    story.append(Paragraph("Comprehensive Project Report", styles["CoverMeta"]))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Prepared for: Soorya", styles["CoverMeta"]))
    story.append(Paragraph("March 2026", styles["CoverMeta"]))
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph("BZ / CY / CZ Pharmaceutical SKU Segments",
                           ParagraphStyle("sub", fontName="Helvetica-Oblique",
                                          fontSize=10, textColor=MED_GRAY,
                                          alignment=TA_CENTER)))
    story.append(PageBreak())

    # ─── Parse markdown ───
    with open("PROJECT_REPORT.md", "r") as f:
        content = f.read()

    lines = content.split("\n")

    # Find start (## 1.)
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith("## 1."):
            start = idx
            break

    i = start
    in_code = False
    code_lines = []
    in_table = False
    table_lines = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if in_code:
                code_text = "\n".join(code_lines)
                story.append(Preformatted(code_text, styles["CodeStyle"]))
                story.append(Spacer(1, 2 * mm))
                code_lines = []
                in_code = False
            else:
                if in_table and table_lines:
                    h, r = parse_md_table(table_lines)
                    if h:
                        story.append(build_table(h, r, styles))
                    table_lines = []
                    in_table = False
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Tables
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            i += 1
            continue
        else:
            if in_table and table_lines:
                h, r = parse_md_table(table_lines)
                if h:
                    story.append(build_table(h, r, styles))
                    story.append(Spacer(1, 2 * mm))
                table_lines = []
                in_table = False

        # H1 (##)
        if line.startswith("## "):
            title = line[3:].strip()
            title = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', title)
            story.append(PageBreak())
            story.append(Paragraph(title, styles["H1"]))
            story.append(HRFlowable(width="40%", thickness=0.8, color=GREEN,
                                     spaceAfter=4 * mm))
            i += 1
            continue

        # H2 (###)
        if line.startswith("### "):
            title = line[4:].strip()
            story.append(Paragraph(clean_md(title), styles["H2"]))
            i += 1
            continue

        # H3 (####)
        if line.startswith("#### "):
            title = line[5:].strip()
            story.append(Paragraph(clean_md(title), styles["H3"]))
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            i += 1
            continue

        # Bullet points
        if line.strip().startswith("- "):
            text = line.strip()[2:].strip()
            text = clean_md(text)
            story.append(Paragraph(text, styles["BulletItem"],
                                   bulletText="\u2022"))
            i += 1
            continue

        # Numbered list
        m = re.match(r'^(\d+)\.\s+(.+)', line.strip())
        if m:
            text = clean_md(m.group(2))
            story.append(Paragraph(text, styles["BulletItem"],
                                   bulletText=f"{m.group(1)}."))
            i += 1
            continue

        # Regular text
        text = line.strip()
        if text:
            text = clean_md(text)
            story.append(Paragraph(text, styles["BodyText2"]))

        i += 1

    # Flush
    if in_table and table_lines:
        h, r = parse_md_table(table_lines)
        if h:
            story.append(build_table(h, r, styles))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print("PDF generated: PROJECT_REPORT.pdf")


if __name__ == "__main__":
    generate_pdf()
