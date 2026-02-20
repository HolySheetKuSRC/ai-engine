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


settings = Settings()
