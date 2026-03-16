from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.user_model import get_user_by_email, create_user
from utils.jwt_handler import create_token

router = APIRouter()

class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "doctor"

@router.post("/api/auth/login")
def login(data: LoginRequest):

    user = get_user_by_email(data.email)

    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # MongoDB uses _id, convert to string id
    user_id = str(user["_id"])

    token = create_token({
        "id": user_id,
        "name": user["name"],
        "role": user["role"]
    })

    return {
        "token": token,
        "user": {
            "id": user_id,
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    }

@router.post("/api/auth/signup")
def signup(data: SignupRequest):
    
    existing_user = get_user_by_email(data.email)
    
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists")
        
    user_id = create_user(data.name, data.email, data.password, data.role)
    
    user = get_user_by_email(data.email)
    
    token = create_token({
        "id": str(user["_id"]),
        "name": user["name"],
        "role": user["role"]
    })
    
    return {
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user["role"]
        }
    }