from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_async_session
from src.schemas.chat import ChatRequest
from src.services.chat_service import process_chat

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_class=StreamingResponse)
async def chat_endpoint(
    request: ChatRequest, 
    db: AsyncSession = Depends(get_async_session)
):
    """
    Chat endpoint supporting RAG context injection and history.
    Streams the response from Typhoon model.
    """
    try:
        response_generator = await process_chat(request, db)
        return StreamingResponse(response_generator, media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
