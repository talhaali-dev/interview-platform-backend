from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/interview_prep"
    
    # Application Settings
    APP_NAME: str = "Interview Prep API"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()
