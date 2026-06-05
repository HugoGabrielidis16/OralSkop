from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from supabase import create_client
from config import settings
from services.db import get_db

router = APIRouter()

# Separate anon client for sign_in (required by Supabase — service role cannot issue user JWTs)
_anon_client = None


def get_anon_client():
    global _anon_client
    if _anon_client is None:
        _anon_client = create_client(settings.supabase_url, settings.supabase_anon_key)
    return _anon_client


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    """
    Uses admin API (service role) to create users — bypasses email rate limits
    and confirmation emails entirely. Correct for a backend-owned auth flow.
    """
    db = get_db()
    try:
        result = db.auth.admin.create_user({
            "email": body.email,
            "password": body.password,
            "email_confirm": True,   # mark as confirmed — no email sent
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.user is None:
        raise HTTPException(status_code=400, detail="Registration failed")

    return {"message": "User created", "user_id": str(result.user.id)}


@router.post("/login")
async def login(body: LoginRequest):
    """
    Uses anon client for sign_in — only the anon key produces a valid user JWT
    that can be decoded with the JWT secret.
    """
    client = get_anon_client()
    try:
        result = client.auth.sign_in_with_password({"email": body.email, "password": body.password})
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    if result.session is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"access_token": result.session.access_token, "token_type": "bearer"}
