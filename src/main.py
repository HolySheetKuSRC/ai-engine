
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.database import init_db
from src.config import settings
from src.routers import ocr, sheets, audio, chat
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown

from src.core.limiter import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

app = FastAPI(title="AI OCR Service", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# อนุญาตให้ Docs Hub ส่ง Request มาหาได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",  # Docs Hub
        "http://localhost:9000",  # Frontend app
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr.router)
app.include_router(sheets.router)
app.include_router(audio.router)
app.include_router(chat.router)

@app.get("/")
def health_check():
    return {"status": "ok", "env": settings.env}

def main() -> None:
    print(f"Service starting in {settings.env} mode")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
