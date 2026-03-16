from database import cases_collection
from datetime import datetime


def create_case(case):
    cases_collection.insert_one(case)


def get_case(case_id):
    return cases_collection.find_one({"case_id": case_id})


def update_prediction(case_id, prediction_data):
    """Update case with full ML prediction results."""
    cases_collection.update_one(
        {"case_id": case_id},
        {"$set": {
            "prediction": prediction_data.get("severity"),
            "confidence": prediction_data.get("confidence"),
            "dr_grade": prediction_data.get("dr_grade"),
            "triage": prediction_data.get("triage"),
            "risk_score": prediction_data.get("risk_score"),
            "heatmap_path": prediction_data.get("heatmap_path"),
            "inference_time_ms": prediction_data.get("inference_time_ms"),
            "status": "In Progress"
        }}
    )


def get_all_cases():
    cases = list(cases_collection.find({}, {"_id": 0}))
    return cases


def resolve_case(case_id):
    """Mark a case as resolved."""
    result = cases_collection.update_one(
        {"case_id": case_id},
        {"$set": {
            "status": "Resolved",
            "resolved_at": datetime.utcnow().isoformat()
        }}
    )
    if result.modified_count == 0:
        return None
    return cases_collection.find_one({"case_id": case_id}, {"_id": 0})


def update_case(case_id, updates):
    """Partial update of a case document."""
    result = cases_collection.update_one(
        {"case_id": case_id},
        {"$set": updates}
    )
    if result.modified_count == 0:
        return None
    return cases_collection.find_one({"case_id": case_id}, {"_id": 0})