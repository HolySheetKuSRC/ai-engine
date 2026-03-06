import os
from datetime import datetime
import json
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, JSON
from uuid import uuid4, UUID

# SQLite Database for Job Tracking
# Use an absolute path so the location is consistent regardless of working directory.
# Defaults to /app/data/jobs.db which is mapped to the host via a Docker volume.
_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "/app/data/jobs.db")
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
DATABASE_URL: str = f"sqlite+aiosqlite:///{_DB_PATH}"

class Base(DeclarativeBase):
    pass

class AnalyzeJob(Base):
    __tablename__ = "analyze_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    sheet_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    filename: Mapped[str | None] = mapped_column(String, nullable=True)
    source_type: Mapped[str] = mapped_column(String, default="audio", nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending") # pending, processing, completed, failed
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

engine = create_async_engine(DATABASE_URL, echo=False)

async_session_maker = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)

async def init_db():
    try:
        # Import models to ensure they are registered with Base.metadata
        from src.models.chat import ChatHistory
        from src.models.ai_dataset import AiDatasetRecord
        from src.models.sales_session import SalesSessionState
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        if "already exists" in str(e):
            pass
        else:
            raise e

async def get_async_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session
