from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.case_model import get_case, update_prediction
from utils.ml_client import run_prediction

router = APIRouter()

class PredictRequest(BaseModel):
    case_id: str


@router.post("/api/predict")
def predict(data: PredictRequest):

    case = get_case(data.case_id)

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    result = run_prediction(case["image_path"])

    if "error" in result:
        raise HTTPException(status_code=502, detail=f"ML API error: {result['error']}")

    # Store the full ML prediction result in the case
    update_prediction(data.case_id, result)

    return {
        "prediction": result.get("severity"),
        "confidence": result.get("confidence"),
        "dr_grade": result.get("dr_grade"),
        "triage": result.get("triage"),
        "risk_score": result.get("risk_score"),
        "heatmap_path": result.get("heatmap_path"),
        "inference_time_ms": result.get("inference_time_ms"),
        "clinical_reasoning": result.get("clinical_reasoning")
    }