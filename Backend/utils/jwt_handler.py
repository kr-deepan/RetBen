import jwt
from config import JWT_SECRET

def create_token(user):
    payload = {
        "id": user["id"],
        "name": user["name"],
        "role": user["role"]
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")