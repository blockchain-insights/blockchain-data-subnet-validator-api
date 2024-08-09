from pydantic_settings import BaseSettings
from pydantic import Field

class ValidatorAPIConfig(BaseSettings):
    API_PORT: int = Field(8001, env="API_PORT")
    DB_CONNECTION_STRING: str = Field("postgresql://postgres:changeit456$@localhost:5433/validator", env="DB_CONNECTION_STRING")
    REDIS_URL: str = Field("redis://localhost:6379", env="REDIS_URL")
    WORKER_COUNT: int = Field(16, env="WORKER_COUNT")
    TIMEOUT: int = Field(60, env="TIMEOUT")
    RATE_LIMIT: int = Field(1024, env="RATE_LIMIT")
    TOP_RATE: float = Field(0.64, env="TOP_RATE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"