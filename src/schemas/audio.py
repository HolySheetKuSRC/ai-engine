from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class AudioHistoryItem(BaseModel):
    job_id: UUID
    filename: Optional[str]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AudioHistoryResponse(BaseModel):
    items: List[AudioHistoryItem]


class AudioResultUpdate(BaseModel):
    result_text: str
