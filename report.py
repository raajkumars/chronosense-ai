"""PDF report generator using reportlab."""
from __future__ import annotations

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_pdf(voice: dict, gait: dict, evaluation: dict, output_path: str = "chronosense_report.pdf") -> str:
    """Generate a single-page diagnostic PDF summary.

    Returns:
        Path to the generated PDF.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                  fontSize=24, spaceAfter=30, textColor=colors.HexColor('#1a1a2e'))
    story.append(Paragraph("ChronoSense AI — Diagnostic Summary", title_style))
    story.append(Spacer(1, 12))

    # Biological Age
    story.append(Paragraph(f"<b>Biological Age Estimate:</b> {evaluation.get('biological_age_estimate', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Frailty Indicator:</b> {evaluation.get('frailty_indicator', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Anomalies
    story.append(Paragraph("<b>Detected Anomalies:</b>", styles['Heading2']))
    for a in evaluation.get("anomalies", []):
        story.append(Paragraph(f"• {a}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Recommendations
    story.append(Paragraph("<b>Agent Recommendations:</b>", styles['Heading2']))
    for r in evaluation.get("recommendations", []):
        story.append(Paragraph(f"• {r}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Voice metrics table
    story.append(Paragraph("<b>Voice Biomarkers:</b>", styles['Heading2']))
    voice_data = [["Metric", "Value"],
                  ["Jitter Proxy", f"{voice.get('jitter_proxy', 0):.4f}"],
                  ["Shimmer Proxy", f"{voice.get('shimmer_proxy', 0):.4f}"],
                  ["Spectral Centroid", f"{voice.get('spectral_centroid_mean', 0):.1f} Hz"],
                  ["Duration", f"{voice.get('duration_s', 0):.1f}s"]]
    t = Table(voice_data, colWidths=[3 * inch, 3 * inch])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                           ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                           ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                           ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Gait metrics table
    story.append(Paragraph("<b>Gait Biomarkers:</b>", styles['Heading2']))
    gait_data = [["Metric", "Value"],
                 ["Asymmetry Index", f"{gait.get('asymmetry_index', 0)}°"],
                 ["Step Frequency", f"{gait.get('step_frequency', 0)} Hz"],
                 ["Frames Processed", str(gait.get('frames_processed', 0))]]
    t2 = Table(gait_data, colWidths=[3 * inch, 3 * inch])
    t2.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    story.append(t2)

    # Disclaimer
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "<i>Disclaimer: This is an experimental demo proxy, not a validated clinical diagnostic. "
        "The Rockwood Frailty Index is used as a reference framework only.</i>",
        styles['Italic']))

    doc.build(story)
    return output_path
