"""
JWT dependency — validates Supabase-issued access tokens.

Supabase now issues ES256 (ECDSA) JWTs for user sessions, not HS256.
Local verification with the JWT secret no longer works for user tokens.
We delegate validation to Supabase via get_user(token) — simpler and always correct.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from services.db import get_db

bearer_scheme = HTTPBearer()


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    token = credentials.credentials
    db = get_db()
    try:
        result = db.auth.get_user(token)
        if result.user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return str(result.user.id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
