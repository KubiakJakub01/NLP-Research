from pathlib import Path

from pydantic import BaseModel


class WhisperHparams(BaseModel):
    model_name: str | Path = 'openai/whisper-base'
    processor_name: str = 'openai/whisper-base'
