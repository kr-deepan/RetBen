from fastapi import APIRouter, HTTPException
from models.case_model import get_all_cases, get_case, resolve_case

router = APIRouter()


@router.get("/api/cases")
def get_cases():
    return get_all_cases()


@router.get("/api/cases/{case_id}")
def case_details(case_id: str):
    case = get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    case["_id"] = str(case["_id"])
    return case


@router.put("/api/cases/{case_id}/resolve")
def resolve(case_id: str):
    case = resolve_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case