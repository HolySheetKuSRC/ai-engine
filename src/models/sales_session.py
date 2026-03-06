from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from src.database import Base


class SalesSessionState(Base):
    """
    Tracks the Brain Audit sales funnel step for each chat session.
    Only used when sheet_id is None (no specific study guide context).
    Steps 1-7 correspond to Sean D'Souza's '7 Red Bags' framework.
    """

    __tablename__ = "sales_session_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    # Current step the bot should respond with next (1–7)
    current_step: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    # The user's stated problem — captured at step 2 to power the RAG hook at step 3
    problem_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
