from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite:///./sortflow.db"
    redis_url: str = "redis://localhost:6379/0"
    workflow_queue: str = "sortflow"
    seed_demo_data: bool = True
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
