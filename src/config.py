from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field

import os


load_dotenv()


class AppConfig(BaseModel):
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    db_path: Path = Field(default_factory=lambda: Path(os.getenv("DB_PATH", "./data/processed/hospital.db")))
    audit_log_path: Path = Field(default_factory=lambda: Path(os.getenv("AUDIT_LOG_PATH", "./logs/audit.log")))


def get_config() -> AppConfig:
    return AppConfig()
