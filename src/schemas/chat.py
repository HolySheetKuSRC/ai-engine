from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID for the chat")
    message: str = Field(..., description="The user's message")
    sheet_id: str | None = Field(None, description="Optional sheet ID for context injection (RAG)")
