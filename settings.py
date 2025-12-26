from pydantic_settings import BaseSettings
from pydantic import Field
import warnings
import os
from huggingface_hub import login as hf_login
from wandb import login as wandb_login


class Settings(BaseSettings):
    openai_api_key: str = Field(...)
    hf_token: str = Field(...)
    hf_username: str = Field(...)
    wandb_api_key: str = Field(...)
    wandb_entity: str = Field(...)
    
    # Optional cache directory configurations
    numba_cache_dir: str | None = Field(None, description="Numba cache directory")
    wandb_dir: str | None = Field(None, description="WandB working directory")
    wandb_cache_dir: str | None = Field(None, description="WandB cache directory")
    wandb_data_dir: str | None = Field(None, description="WandB data directory")
    hf_home: str | None = Field(None, description="Hugging Face cache home")
    transformers_cache: str | None = Field(None, description="Transformers cache directory")

    class Config:
        env_file = '.env'
        extra = "allow"
    
    def model_post_init(self, __context) -> None:
        """Automatically login and configure cache directories after settings initialization."""
        # Set cache directories if specified
        if self.numba_cache_dir:
            os.environ.setdefault('NUMBA_CACHE_DIR', self.numba_cache_dir)
        if self.wandb_dir:
            os.environ.setdefault('WANDB_DIR', self.wandb_dir)
        if self.wandb_cache_dir:
            os.environ.setdefault('WANDB_CACHE_DIR', self.wandb_cache_dir)
        if self.wandb_data_dir:
            os.environ.setdefault('WANDB_DATA_DIR', self.wandb_data_dir)
        if self.hf_home:
            os.environ.setdefault('HF_HOME', self.hf_home)
        if self.transformers_cache:
            os.environ.setdefault('TRANSFORMERS_CACHE', self.transformers_cache)
        
        # Login to services
        try:
            hf_login(token=self.hf_token)
        except Exception as e:
            warnings.warn(f"Failed to login to Hugging Face Hub: {e}")
        
        try:
            wandb_login(key=self.wandb_api_key)
        except Exception as e:
            warnings.warn(f"Failed to login to Weights & Biases: {e}")

settings = Settings()
