from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from src.database import Base

class AiDatasetRecord(Base):
    __tablename__ = "ai_dataset_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String, default='audio', nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=True)         # Full OCR content
    summary_text: Mapped[str] = mapped_column(Text, nullable=True)     # AI-generated summary
    # Comma-separated tags produced by AI analysis, e.g. "SVM, machine learning, classification"
    # Stored as Text so LIKE queries work without a JSON extension.
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
