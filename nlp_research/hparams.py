'''Module for hyperparameters.'''
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HParams:
    '''Base class with hyperparameters.'''
    # Data
    train_data_dir: list[Path] = field(
        default_factory=list, metadata={'help': 'Path to training data.'}
    )
    valid_data_dir: list[Path] = field(
        default_factory=list, metadata={'help': 'Path to validation data.'}
    )
    checkpoint_dir: Path = field(
        default=Path('models/checkpoints'),
        metadata={'help': 'Directory to save checkpoints'},
    )
    log_dir: Path = field(
        default=Path('models/logs'),
        metadata={'help': 'Directory to save tensorboard logs'},
    )
