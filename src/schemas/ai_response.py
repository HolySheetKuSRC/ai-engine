
from pydantic import BaseModel, Field
from typing import List, Optional

class AIAnalysisResult(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    ocr_content: str = Field(..., description="Full text extracted from OCR")
    summary: str = Field(..., description="AI-generated summary of the content")
    assessment: List[str] = Field(..., description="List of key assessment points")
    tags: List[str] = Field(..., description="List of relevant hashtags")
    suggested_price: float = Field(0.0, description="Suggested price based on content quality")
    page_count: Optional[int] = Field(None, description="Number of pages in the document")
