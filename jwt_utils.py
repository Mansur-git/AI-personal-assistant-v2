from datetime import datetime, timedelta,timezone 
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from bson import ObjectId
from mongo import Users

SECRET_KEY = "dev-secret-change-later"
ALGORITHM = "HS256"

def create_access_token(user_id: str):
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        if not await Users.is_valid_user_id(user_id):
            raise HTTPException(status_code=401, detail="User not found")

        return user_id

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
