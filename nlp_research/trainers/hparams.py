from pathlib import Path

from pydantic import BaseModel


class WhisperHparams(BaseModel):
    # Model
    model_name: str | Path = 'openai/whisper-base'
    processor_name: str = 'openai/whisper-base'

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

    # Training
    output_dir: Path = Path('output')
    resume_from_checkpoint: str | Path | None = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    warmup_steps: int = 0
    max_steps: int = 10_000
    gradient_checkpointing: bool = False
    use_fp16: bool = False
    evaluation_strategy: str = 'steps'
    eval_batch_size: int = 8
    predict_with_generate: bool = True
    max_target_length: int = 128
    save_steps: int = 500
    logging_steps: int = 100
    push_to_hub: bool = False
