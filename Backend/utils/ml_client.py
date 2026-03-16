import requests
from config import ML_API_URL


def run_prediction(image_path):
    """Send image to ML API and return the full prediction result."""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.split("/")[-1], f, "image/jpeg")}
            res = requests.post(
                f"{ML_API_URL}/predict",
                files=files,
                timeout=60
            )

        if res.status_code != 200:
            return {"error": f"ML API returned status {res.status_code}: {res.text}"}

        return res.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to ML API. Ensure it is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}