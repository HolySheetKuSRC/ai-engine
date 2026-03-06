from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_async_session
from src.schemas.chat import ChatRequest
from src.services.chat_service import process_chat
from src.services.chat_service import _get_or_create_sales_state
from src.models.sales_session import SalesSessionState
from sqlalchemy import select, delete
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
    - With sheet_id: Tutor Mode (answers questions about a purchased sheet).
    - Without sheet_id: Brain Audit Sales Bot Mode (7-step consultative funnel).
    Returns a unified JSON response.
    """
    try:
        response_data = await process_chat(chat_request, db)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sales-session/{session_id}")
async def reset_sales_session(
    session_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Reset the Brain Audit sales funnel for a given session back to Step 1.
    Useful for testing or when a user wants to restart the consultation.
    """
    stmt = delete(SalesSessionState).where(SalesSessionState.session_id == session_id)
    await db.execute(stmt)
    await db.commit()
    return {"session_id": session_id, "message": "Sales session reset to Step 1."}


@router.get("/sales-session/{session_id}")
async def get_sales_session(
    session_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Inspect the current Brain Audit step for a session (for debugging / frontend state).
    """
    stmt = select(SalesSessionState).where(SalesSessionState.session_id == session_id)
    result = await db.execute(stmt)
    state = result.scalar_one_or_none()
    if state is None:
        return {"session_id": session_id, "current_step": 1, "problem_text": None}
    return {
        "session_id": session_id,
        "current_step": state.current_step,
        "problem_text": state.problem_text,
    }
