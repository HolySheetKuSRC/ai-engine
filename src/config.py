from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    env: str = os.getenv("APP_ENV", "development")
    TYPHOON_API_KEY: str = os.getenv("TYPHOON_API_KEY")
    TYPHOON_BASE_URL: str = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    # SQLite path — must match the Docker volume mount target in docker-compose.yml
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "/app/data/jobs.db")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "changeme")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")


settings = Settings()
