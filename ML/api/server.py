import os
import shutil
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time

from models.efficientnet_model import load_model
from inference.predict import predict_image
from explainability.gradcam import generate_gradcam
from triage.triage_logic import get_triage

app = FastAPI(title="DR Screening API", description="Diabetic Retinopathy Screening Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model and device
MODEL = None
DEVICE = None
CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    global MODEL, DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting API server.")
    print(f"Using device: {DEVICE}")
    
    # We load the model once at startup for Phase 9 optimization
    if os.path.exists(CHECKPOINT_PATH):
        try:
            MODEL = load_model(CHECKPOINT_PATH, DEVICE)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Ensure the model is trained before inference.")

@app.post("/predict")
async def predict_dr(file: UploadFile = File(...)):
    if MODEL is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded. Train the model first."})
        
    start_time = time.time()
    
    # Complete Phase 10 File saved locally
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Phase 5: Inference (which includes Phase 2 Preprocessing)
        prediction_result = predict_image(file_location, MODEL, DEVICE)
        
        if "error" in prediction_result:
            return JSONResponse(status_code=500, content={"error": prediction_result["error"]})
            
        dr_grade = prediction_result["class"]
        severity = prediction_result["severity"]
        confidence = prediction_result["confidence"]
        
        # Phase 7: Triage
        triage_result = get_triage(dr_grade, confidence)
        
        # Phase 6: GradCAM
        heatmap_path = generate_gradcam(file_location, MODEL, target_class=dr_grade, output_dir=OUTPUT_DIR, device=DEVICE)
        
        # Format response to match requirements exactly
        response = {
            "dr_grade": dr_grade,
            "severity": severity,
            "confidence": confidence,
            "risk_score": triage_result["risk_score"],
            "triage": triage_result["triage"],
            "heatmap_path": heatmap_path.replace("\\", "/"),
            "inference_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return response
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/heatmap/{filename}")
async def get_heatmap(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "Heatmap not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
