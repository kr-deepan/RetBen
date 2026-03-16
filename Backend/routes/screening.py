from fastapi import APIRouter, UploadFile, File, Form
import uuid
import shutil
from datetime import date
from models.case_model import create_case

router = APIRouter()

UPLOAD_FOLDER = "uploads"

@router.post("/api/screening/upload")
async def upload_image(
    image: UploadFile = File(...),
    patient_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...)
):

    case_id = "DR" + str(uuid.uuid4())[:6]

    path = f"{UPLOAD_FOLDER}/{case_id}.jpg"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    case = {
        "case_id": case_id,
        "patient_name": patient_name,
        "age": age,
        "gender": gender,
        "image_path": path,
        "prediction": None,
        "confidence": None,
        "status": "Pending",
        "date": str(date.today())
    }

    create_case(case)

    return {
        "case_id": case_id,
        "message": "Image uploaded successfully"
    }