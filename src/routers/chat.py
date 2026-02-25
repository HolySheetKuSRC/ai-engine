from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_async_session
from src.schemas.chat import ChatRequest
from src.services.chat_service import process_chat
from src.core.limiter import limiter

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
    responses={404: {"description": "Not found"}},
)

@router.post("/")
@limiter.limit("10/minute")
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest, 
    db: AsyncSession = Depends(get_async_session)
):
    """
    Chat endpoint supporting RAG context injection and history.
    Returns a unified JSON response.
    """
    try:
        response_data = await process_chat(chat_request, db)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
