import os
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def ensure_export_dir() -> str:
    export_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


def generate_report_pdf(test_doc: Dict[str, Any], user: Dict[str, Any]) -> str:
    export_dir = ensure_export_dir()
    filename = f"CardioClue_Report_{test_doc['testId']}.pdf"
    path = os.path.join(export_dir, filename)

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    margin = 20 * mm
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "CardioClue - Cardiovascular Risk Report")
    y -= 12 * mm

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"User: {user.get('name')}  |  ID: {user.get('userId')}")
    y -= 6 * mm
    c.drawString(margin, y, f"Age: {user.get('age', '-')}, Gender: {user.get('gender', '-')}")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, f"Risk Result: {test_doc.get('riskResult')}")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Test ID: {test_doc.get('testId')}  |  Date: {test_doc.get('timestamp').strftime('%Y-%m-%d %H:%M') if test_doc.get('timestamp') else ''}")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Responses:")
    y -= 7 * mm
    c.setFont("Helvetica", 10)
    for key, val in test_doc.get('responses', {}).items():
        c.drawString(margin, y, f"- {key}: {val}")
        y -= 6 * mm
        if y < margin:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return path


