import os
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import cases_collection
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

router = APIRouter()

class ReportRequest(BaseModel):
    case_id: str

@router.post("/api/report/generate")
def generate_report(data: ReportRequest):
    case = cases_collection.find_one({"case_id": data.case_id})

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    path = f"reports/{data.case_id}.pdf"
    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate(
        path, 
        pagesize=letter,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch
    )

    # Define custom aesthetic colors
    PRIMARY_COLOR = colors.HexColor("#0f172a") # Slate 900
    SECONDARY_COLOR = colors.HexColor("#334155") # Slate 700
    ACCENT_COLOR = colors.HexColor("#3b82f6") # Blue 500
    BG_COLOR = colors.HexColor("#f8fafc") # Slate 50
    BORDER_COLOR = colors.HexColor("#e2e8f0") # Slate 200

    styles = getSampleStyleSheet()
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=PRIMARY_COLOR,
        spaceAfter=20,
        alignment=1 # Center
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=PRIMARY_COLOR,
        spaceBefore=15,
        spaceAfter=10,
        borderPadding=5,
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=SECONDARY_COLOR,
        leading=14
    )

    content = []

    # Document Header
    report_date = datetime.now().strftime("%B %d, %Y")
    content.append(Paragraph("<b>DIABETIC RETINOPATHY SCREENING REPORT</b>", title_style))
    content.append(Paragraph(f"<font color='#64748b'>Report Generated: {report_date}</font>", ParagraphStyle('date', parent=normal_style, alignment=1)))
    content.append(Spacer(1, 0.4 * inch))

    # Patient Information Table
    content.append(Paragraph("Patient Information", section_style))
    
    patient_data = [
        [Paragraph("<b>Case ID:</b>", normal_style), Paragraph(data.case_id, normal_style), Paragraph("<b>Status:</b>", normal_style), Paragraph(case.get("status", "Pending"), normal_style)],
        [Paragraph("<b>Patient Name:</b>", normal_style), Paragraph(case.get("patient_name", "N/A"), normal_style), Paragraph("<b>Date of Birth/Age:</b>", normal_style), Paragraph(str(case.get("age", "N/A")), normal_style)],
        [Paragraph("<b>Gender:</b>", normal_style), Paragraph(case.get("gender", "N/A"), normal_style), "", ""]
    ]
    
    patient_table = Table(patient_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_COLOR),
        ('BOX', (0,0), (-1,-1), 1, BORDER_COLOR),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    content.append(patient_table)
    content.append(Spacer(1, 0.3 * inch))

    # Analysis Results Section
    content.append(Paragraph("AI Analysis Results", section_style))
    
    confidence = case.get('confidence')
    conf_str = f"{(confidence * 100):.1f}%" if confidence else 'N/A'
    
    risk = case.get('risk_score')
    risk_str = f"{(risk * 100):.1f}%" if risk else 'N/A'
    
    severity = case.get("prediction", "Pending")
    
    metrics_data = [
        ["Diagnosis Priority:", "AI Assessment:"],
        [case.get("triage", "N/A").upper(), severity.upper()],
        ["Confidence Score:", f"Risk Score:"],
        [conf_str, risk_str]
    ]

    metrics_table = Table(metrics_data, colWidths=[3.25*inch, 3.25*inch])
    
    # Color logic based on severity/triage
    triage_color = colors.HexColor("#22c55e") # Green
    if case.get("triage") == "Urgent":
        triage_color = colors.HexColor("#ef4444") # Red
        
    sev_color = colors.HexColor("#3b82f6") # Blue
    if severity in ["Severe DR", "Proliferative DR"]:
        sev_color = colors.HexColor("#ef4444")

    metrics_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        # Header Row
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,0), SECONDARY_COLOR),
        ('BOTTOMPADDING', (0,0), (-1,0), 2),
        # Value Row 1
        ('FONTNAME', (0,1), (0,1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,1), (0,1), triage_color),
        ('FONTSIZE', (0,1), (-1,1), 16),
        ('FONTNAME', (1,1), (1,1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1,1), (1,1), sev_color),
        ('BOTTOMPADDING', (0,1), (-1,1), 15),
        # Header Row 2
        ('FONTNAME', (0,2), (-1,2), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,2), (-1,2), SECONDARY_COLOR),
        ('BOTTOMPADDING', (0,2), (-1,2), 2),
        # Value Row 2
        ('FONTSIZE', (0,3), (-1,3), 14),
        ('TEXTCOLOR', (0,3), (-1,3), PRIMARY_COLOR),
    ]))
    content.append(metrics_table)
    content.append(Spacer(1, 0.3 * inch))

    # Imagery Section
    content.append(Paragraph("Clinical Imaging", section_style))

    img_elements = []
    
    # Original Image
    img_path = case.get("image_path")
    if img_path and os.path.exists(img_path):
        try:
            original_img = Image(img_path, width=3.1*inch, height=3.1*inch, kind='proportional')
            img_elements.append([original_img])
            img_elements.append([Paragraph("<font color='#334155'><b>Original Fundus Photograph</b></font>", ParagraphStyle('c', alignment=1))])
        except Exception as e:
            pass

    # Heatmap Image
    heatmap_path_raw = case.get("heatmap_path")
    if heatmap_path_raw:
        backend_cwd = os.getcwd()
        full_heatmap_path = os.path.join(os.path.dirname(backend_cwd), "ML", heatmap_path_raw)
        
        if os.path.exists(full_heatmap_path):
            try:
                heatmap_img = Image(full_heatmap_path, width=3.1*inch, height=3.1*inch, kind='proportional')
                if len(img_elements) == 2:
                    img_elements[0].append(heatmap_img)
                    img_elements[1].append(Paragraph("<font color='#334155'><b>AI Attention Heatmap (Grad-CAM)</b></font>", ParagraphStyle('c', alignment=1)))
                else:
                    img_elements.append([heatmap_img])
                    img_elements.append([Paragraph("<font color='#334155'><b>AI Attention Heatmap (Grad-CAM)</b></font>", ParagraphStyle('c', alignment=1))])
            except Exception as e:
                pass

    if img_elements:
        img_table = Table(img_elements, colWidths=[3.25*inch] * len(img_elements[0]))
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10)
        ]))
        content.append(img_table)
    else:
        content.append(Paragraph("<i>No imaging available for this report.</i>", normal_style))

    # Clinical Reasoning Section
    clinical_reasoning = case.get("clinical_reasoning")
    if clinical_reasoning:
        content.append(Spacer(1, 0.3 * inch))
        content.append(Paragraph("Clinical Reasoning", section_style))
        for paragraph in clinical_reasoning.split("\n\n"):
            if paragraph.strip():
                content.append(Paragraph(paragraph, normal_style))
                content.append(Spacer(1, 0.1 * inch))

    # Footer
    content.append(Spacer(1, 0.5 * inch))
    content.append(Paragraph("<font size=8 color='#94a3b8'>This report was generated automatically by the AI screening system. Clinical decisions should always be verified by a qualified ophthalmologist.</font>", ParagraphStyle('footer', alignment=1)))

    doc.build(content)

    return {
        "report_url": f"/reports/{data.case_id}.pdf"
    }