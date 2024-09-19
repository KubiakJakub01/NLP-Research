from pathlib import Path

from pydantic import BaseModel


class WhisperHparams(BaseModel):
    # Model
    model_name: str | Path = 'openai/whisper-base'
    processor_name: str = 'openai/whisper-base'
    output_dir: Path = Path('output')
    resume_from_checkpoint: str | Path | None = None

    # Data
    dataset_name: str = 'mozilla-foundation/common_voice_13_0'
    subset: str = 'pl'
    remove_columns: list[str] = [
        'accent',
        'age',
        'client_id',
        'down_votes',
        'gender',
        'locale',
        'path',
        'segment',
        'up_votes',
    ]
    sample_rate: int = 16_000
