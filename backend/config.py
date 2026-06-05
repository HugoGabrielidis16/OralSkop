from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_jwt_secret: str

    # AWS S3
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str = ""
    aws_s3_region: str = "us-west-2"
    s3_bucket_photos: str = "oralskop-photos"
    s3_bucket_masked: str = "oralskop-masked"

    # AI
    ai_server_url: str = "http://localhost:8001/analyze"
    use_mock_ai: bool = True

    # App
    environment: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
