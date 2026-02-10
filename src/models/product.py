
import uuid
import enum
import datetime
from sqlalchemy import String, Boolean, Integer, Numeric, Text, ForeignKey, TIMESTAMP, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from src.database import Base

class SheetStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class University(Base):
    """
    University model mapped to the 'universities' table.
    """
    __tablename__ = "universities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name_th: Mapped[str | None] = mapped_column(String, nullable=True)
    name_en: Mapped[str | None] = mapped_column(String, nullable=True)

    sheets: Mapped[list["Product"]] = relationship(back_populates="university")

class Product(Base):
    """
    Product model mapped to the 'sheets' table in the Mock DB schema.
    """
    __tablename__ = "sheets"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    seller_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    
    university_id: Mapped[int | None] = mapped_column(ForeignKey("universities.id"), nullable=True)
    category_id: Mapped[int | None] = mapped_column(Integer, nullable=True) # Mocking category_id as int for now

    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    price: Mapped[float] = mapped_column(Numeric, default=0, nullable=False)
    file_url: Mapped[str] = mapped_column(String, nullable=False)
    
    # AI Analysis Fields
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_assessment: Mapped[dict | None] = mapped_column(Text, nullable=True) # Storing JSON as Text for simplicity in mock
    tags: Mapped[list[str] | None] = mapped_column(Text, nullable=True) # Storing JSON list as Text
    ocr_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    status: Mapped[SheetStatus] = mapped_column(String, default=SheetStatus.PENDING, nullable=False)
    is_published: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    created_at: Mapped[datetime.datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    updated_at: Mapped[datetime.datetime | None] = mapped_column(TIMESTAMP, onupdate=func.now(), nullable=True)
    
    # Relationship
    university: Mapped["University"] = relationship(back_populates="sheets")

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, title={self.title}, price={self.price})>"
