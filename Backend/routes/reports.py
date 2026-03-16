from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import cases_collection
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

router = APIRouter()

class ReportRequest(BaseModel):
    case_id: str

@router.post("/api/report/generate")
def generate_report(data: ReportRequest):

    case = cases_collection.find_one({"case_id": data.case_id})

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    path = f"reports/{data.case_id}.pdf"

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(path)

    content = [
        Paragraph(f"<b>DR Screening Report</b>", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>Case ID:</b> {data.case_id}", styles["Normal"]),
        Paragraph(f"<b>Patient:</b> {case.get('patient_name', 'N/A')}", styles["Normal"]),
        Paragraph(f"<b>Age:</b> {case.get('age', 'N/A')}", styles["Normal"]),
        Paragraph(f"<b>Gender:</b> {case.get('gender', 'N/A')}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>Prediction:</b> {case.get('prediction', 'Pending')}", styles["Normal"]),
        Paragraph(f"<b>Confidence:</b> {case.get('confidence', 'N/A')}", styles["Normal"]),
        Paragraph(f"<b>Triage:</b> {case.get('triage', 'N/A')}", styles["Normal"]),
        Paragraph(f"<b>Risk Score:</b> {case.get('risk_score', 'N/A')}", styles["Normal"]),
    ]

    doc.build(content)

    return {
        "report_url": f"/reports/{data.case_id}.pdf"
    }