"""Configuration settings for Scholar AI Agent."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application configuration settings."""
    
    # API Configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # File Storage
    upload_dir: str = "uploads"
    max_file_size_mb: int = 50
    allowed_extensions: tuple = (".pdf",)
    
    # LLM Settings
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    llm_timeout_seconds: int = 30
    
    # Google Scholar Settings
    scholar_max_results: int = 20
    scholar_year_filter: int = 2  # Last N years
    scholar_retry_attempts: int = 3
    scholar_retry_delay: float = 1.0
    
    # Performance Settings
    pdf_parse_timeout: int = 5
    total_workflow_timeout: int = 60
    cache_ttl_hours: int = 24
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir, exist_ok=True)