from pydantic_settings import BaseSettings
from pydantic import Field
import warnings
from huggingface_hub import login as hf_login
from wandb import login as wandb_login


class Settings(BaseSettings):
    openai_api_key: str = Field(...)
    hf_token: str = Field(...)
    hf_username: str = Field(...)
    wandb_api_key: str = Field(...)
    wandb_entity: str = Field(...)

    class Config:
        env_file = '.env'
        extra = "allow"
    
    def model_post_init(self, __context) -> None:
        """Automatically login to Hugging Face Hub and Weights & Biases after settings initialization."""
        try:
            hf_login(token=self.hf_token)
        except Exception as e:
            warnings.warn(f"Failed to login to Hugging Face Hub: {e}")
        
        try:
            wandb_login(key=self.wandb_api_key)
        except Exception as e:
            warnings.warn(f"Failed to login to Weights & Biases: {e}")

settings = Settings()
