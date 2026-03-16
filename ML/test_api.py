from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)

def test_predict():
    with open('dataset/gaussian_filtered_images/No_DR/002c21358ce6.png', 'rb') as f:
        # FastAPI triggers the lifespan startup event manually? 
        # TestClient handles startup events in a `with` block context manager.
        pass

if __name__ == "__main__":
    with TestClient(app) as client:
        with open('dataset/gaussian_filtered_images/No_DR/002c21358ce6.png', 'rb') as f:
            response = client.post("/predict", files={"file": ("002c21358ce6.png", f, "image/png")})
            print(response.json())
