"""
CreditLens AI - PDF Report Generator
======================================
Generates professional educational PDF reports using fpdf2.

Includes:
  - Header with project name and date
  - Applicant information table
  - Prediction results
  - Top feature contributions
  - Model performance summary
  - Educational disclaimer
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORTS_DIR


def _safe_str(value):
    """Convert value to a safe ASCII-friendly string for PDF."""
    s = str(value)
    # Replace problematic Unicode characters
    replacements = {
        '\u2265': '>=',  # ≥
        '\u2264': '<=',  # ≤
        '\u2013': '-',   # en dash
        '\u2014': '--',  # em dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...',  # ellipsis
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    # Remove any remaining non-latin1 characters
    s = s.encode('latin-1', errors='replace').decode('latin-1')
    return s


def generate_pdf_report(applicant_data, prediction_results,
                        top_features=None, metrics=None, model_name='XGBoost'):
    """
    Generate a professional PDF report.

    Args:
        applicant_data: dict of {display_name: display_value}
        prediction_results: dict with approval_probability, default_risk, risk_class
        top_features: list of dicts with feature, direction, value
        metrics: dict of model metrics (optional)
        model_name: name of the model used

    Returns:
        bytes: PDF content
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ---- Header ----
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'CreditLens AI', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, 'Explainable Credit Risk Analysis Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # ---- Date and Model ----
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 100, 100)
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    pdf.cell(0, 6, f'Generated: {date_str}  |  Model: {_safe_str(model_name)}',
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='R')
    pdf.ln(5)

    # ---- Prediction Results ----
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Prediction Results', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(79, 195, 247)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    risk_class = _safe_str(prediction_results.get('risk_class', 'Unknown'))
    approval = prediction_results.get('approval_probability', 0)
    default = prediction_results.get('default_risk', 0)

    # Risk class color
    if 'Low' in risk_class:
        pdf.set_fill_color(200, 230, 201)
        text_color = (27, 94, 32)
    elif 'High' in risk_class:
        pdf.set_fill_color(255, 205, 210)
        text_color = (183, 28, 28)
    else:
        pdf.set_fill_color(255, 236, 179)
        text_color = (230, 81, 0)

    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(*text_color)
    pdf.cell(0, 12, f'  {risk_class}', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(3)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(95, 8, f'Approval Probability: {approval:.1%}', new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(95, 8, f'Default Risk: {default:.1%}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(95, 8, f'Threshold: 50% (fixed)', new_x=XPos.RIGHT, new_y=YPos.TOP)
    approved_str = 'Yes' if prediction_results.get('approved', False) else 'No'
    pdf.cell(95, 8, f'Meets Threshold: {approved_str}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # ---- Applicant Information ----
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Applicant Information', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(79, 195, 247)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font('Helvetica', '', 9)
    if applicant_data:
        # Table header
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(85, 7, '  Feature', border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(105, 7, '  Value', border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font('Helvetica', '', 9)
        for key, value in applicant_data.items():
            safe_key = _safe_str(key)
            safe_val = _safe_str(value)
            # Truncate long values
            if len(safe_val) > 50:
                safe_val = safe_val[:47] + '...'
            pdf.cell(85, 6, f'  {safe_key}', border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(105, 6, f'  {safe_val}', border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)

    # ---- Top Contributing Features ----
    if top_features:
        # Check if we need a new page
        if pdf.get_y() > 220:
            pdf.add_page()

        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Top Contributing Features', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(79, 195, 247)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(80, 7, '  Feature', border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(60, 7, '  Direction', border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(50, 7, '  Impact', border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font('Helvetica', '', 9)
        for feat in top_features[:7]:
            name = _safe_str(feat.get('feature', ''))
            direction = _safe_str(feat.get('direction', ''))
            value = feat.get('value', 0)
            if len(name) > 35:
                name = name[:32] + '...'
            pdf.cell(80, 6, f'  {name}', border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(60, 6, f'  {direction}', border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.cell(50, 6, f'  {value:.4f}', border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)

    # ---- Model Performance ----
    if metrics:
        if pdf.get_y() > 230:
            pdf.add_page()

        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Model Performance (Held-out Test Set)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(79, 195, 247)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_font('Helvetica', '', 10)
        for metric_name, metric_val in metrics.items():
            pdf.cell(60, 7, f'  {_safe_str(metric_name)}:')
            if isinstance(metric_val, float):
                pdf.cell(60, 7, f'{metric_val:.4f}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(60, 7, _safe_str(str(metric_val)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ---- Disclaimer ----
    pdf.ln(10)
    if pdf.get_y() > 245:
        pdf.add_page()

    pdf.set_fill_color(255, 243, 224)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 8, '  Educational Disclaimer', new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 100, 100)
    disclaimer = (
        'This report is generated by CreditLens AI, an educational prototype. '
        'It is NOT a real credit assessment. The predictions are based on a historical '
        'German dataset (1994, 1000 records) and should not be used for actual lending decisions. '
        'The model has known limitations including potential biases from historical data, '
        'a fixed 50% classification threshold, and no formal fairness evaluation.'
    )
    pdf.multi_cell(0, 5, disclaimer)

    # Return as bytes
    return bytes(pdf.output())


def save_pdf_report(pdf_bytes, filename=None):
    """Save PDF report to the reports directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'credit_report_{timestamp}.pdf'

    path = os.path.join(REPORTS_DIR, filename)
    with open(path, 'wb') as f:
        f.write(pdf_bytes)

    return path
