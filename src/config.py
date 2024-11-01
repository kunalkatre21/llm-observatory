from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    openrouter_api_key: str = Field(..., env='OPENROUTER_API_KEY')
    default_model: str = "nousresearch/hermes-3-llama-3.1-405b:free"
    max_retries: int = 3
    timeout: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False
