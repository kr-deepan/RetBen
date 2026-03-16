# Diabetic Retinopathy API Documentation

This document serves as the integration interface reference for Frontend and Backend developers connecting to the Diabetic Retinopathy (DR) Screening ML Pipeline.

---

## 1. Server Configuration

The API is built using **FastAPI**. It leverages Uvicorn as the ASGI server. 

### Starting the Server
To run the server locally on your machine, navigate to the project directory and execute:
```bash
py -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```
* **Host `0.0.0.0`**: Binds the server to all available network interfaces (Localhost and Local LAN IP).
* **Port `8000`**: Default HTTP port.

### Interactive API Explorer
FastAPI provides a built-in Swagger UI dashboard for testing the API directly in your browser without writing code.
* **URL**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 2. Endpoints

### 2.1 Predict DR Severity
Runs a fundus image through the pre-loaded EfficientNet-B0 ML model, calculates the diagnostic severity, generates clinical triage rules, and outputs a GradCAM visual heatmap.

* **URL**: `/predict`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

#### Request Payload
The endpoint expects a single file upload parameter.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | `binary` / `File` | Yes | The raw image file of the retinal fundus (PNG, JPG, JPEG). The system dynamically normalizes dimensions during preprocessing, so arbitrary resolutions are supported. |

#### Success Response
* **Code**: `200 OK`
* **Format**: `application/json`

**Example Response:**
```json
{
  "dr_grade": 2,
  "severity": "Moderate DR",
  "confidence": 0.8242,
  "risk_score": 0.8242,
  "triage": "Refer to specialist",
  "heatmap_path": "output/heatmap_002c21358ce6.png",
  "inference_time_ms": 110.45
}
```

#### Fields Description:
* `dr_grade` *(integer)*: The raw numerical class prediction (0 through 4).
* `severity` *(string)*: The human-readable string mapping of the grade (e.g., "No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR").
* `confidence` *(float)*: The softmax probability certainty of the prediction spanning `0.0` to `1.0`.
* `risk_score` *(float)*: Clinicial severity scalar. Mapped logically based on the confidence bounds and grade limits.
* [triage](file:///e:/Projects/ETE/ML/triage/triage_logic.py#1-33) *(string)*: Clinical action required. Either `"Monitor"` (for grades 0-1) or `"Refer to specialist"` (for grades 2-4).
* `heatmap_path` *(string)*: The relative path to the generated GradCAM explanation image saved on the server's local storage.
* `inference_time_ms` *(float)*: The total round-trip latency of the actual preprocessing and GPU tensor calculation process in milliseconds.

#### Error Responses
* **Code `500 Internal Server Error`**: Usually triggered if the image file is fundamentally corrupted or OpenCV fails to read its buffer array.
  ```json
  { "error": "Detailed python exception string" }
  ```
* **Code `503 Service Unavailable`**: Triggered if a POST request hits the endpoint but the system could not locate the ML checkpoint (`best_model.pth`) during startup.

---

## 3. Retrieve GradCAM Heatmap
Once a prediction is successfully made, the server generates an explainability heatmap locally. The frontend must fetch this image sequentially to display it to the user.

* **URL**: `/heatmap/{filename}`
* **Method**: `GET`

#### Request Path Variables
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filename` | `string` | Yes | The exact basename of the heatmap file (e.g., `heatmap_002c21358ce6.png`). You should dynamically copy this from the `heatmap_path` JSON field of the `/predict` response (by splitting off the `output/` prefix). |

#### Success Response
* **Code**: `200 OK`
* **Content-Type**: `image/jpeg` / `image/png` (Raw Binary Image File)

#### Error Responses
* **Code `404 Not Found`**: Triggered if the heatmap file has been deleted or requested before prediction completion.
  ```json
  { "error": "Heatmap not found" }
  ```

---

## 4. Example Integration (cURL)

**1. Submit Image for Prediction:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@patient_eye_scan.png;type=image/png'
```

**2. Fetch the newly computed Heatmap (using the filename returned above):**
```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/heatmap/heatmap_patient_eye_scan.png' \
  --output my_local_diagnostic_scan.png
```
