import logging
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import settings

logger = logging.getLogger(__name__)

_security = HTTPBearer()


@dataclass
class CurrentUser:
    id: str


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> CurrentUser:
    """Decode the incoming Bearer JWT and return the authenticated user's id."""
    token = credentials.credentials
    logger.info(f"Received token starting with: {token[:10]}...")

    try:
        payload: dict = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        logger.error("JWT Validation Failed: Token has expired.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidSignatureError:
        logger.error(
            "JWT Validation Failed: Signature mismatch. "
            "Check if JWT_SECRET matches the auth server."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Signature verification failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.DecodeError as e:
        logger.error(f"JWT Validation Failed: Malformed token. Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"JWT Validation Failed: Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str | None = payload.get("sub")
    if not user_id:
        logger.error(
            f"JWT Validation Failed: 'sub' claim not found in payload. "
            f"Available claims: {list(payload.keys())}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"JWT validated successfully for user_id: {user_id}")
    return CurrentUser(id=user_id)
