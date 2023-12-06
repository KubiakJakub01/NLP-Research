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


@dataclass
class GPTHparams(HParams):
    '''Hyperparameters for GPT.'''
    # Model
    n_embd: int = field(
        default=768, metadata={'help': 'Size of embedding layer.'}
    )
    n_layer: int = field(
        default=12, metadata={'help': 'Number of transformer layers.'}
    )
    n_head: int = field(
        default=12, metadata={'help': 'Number of attention heads.'}
    )
    latent_size: int = field(
        default=0, metadata={'help': 'Size of latent vector.'}
    )
    n_ctx: int = field(
        default=1024, metadata={'help': 'Size of context window.'}
    )
    vocab_size: int = field(
        default=50257, metadata={'help': 'Size of vocabulary.'}
    )
    n_positions: int = field(
        default=1024, metadata={'help': 'Number of positional embeddings.'}
    )
    activation_function: str = field(
        default='gelu', metadata={'help': 'Activation function to use.'}
    )
    resid_pdrop: float = field(
        default=0.1, metadata={'help': 'Dropout probability for residual connections.'}
    )
    embd_pdrop: float = field(
        default=0.1, metadata={'help': 'Dropout probability for embedding layer.'}
    )
    attn_pdrop: float = field(
        default=0.1, metadata={'help': 'Dropout probability for attention layers.'}
    )
    layer_norm_epsilon: float = field(
        default=1e-5, metadata={'help': 'Epsilon for layer normalization.'}
    )
    initializer_range: float = field(
        default=0.02, metadata={'help': 'Range for random initialization.'}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={'help': 'Whether to use gradient checkpointing.'}
    )
    # Training
    batch_size: int = field(
        default=8, metadata={'help': 'Batch size for training.'}
    )
    learning_rate: float = field(
        default=6.25e-5, metadata={'help': 'Learning rate for training.'}
    )
    weight_decay: float = field(
        default=0.01, metadata={'help': 'Weight decay for training.'}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={'help': 'Epsilon for Adam optimizer.'}
    )


@dataclass
class UNetHparams(HParams):
    '''Hyperparameters for UNet.'''
    # Model
    in_channels: int = field(
        default=3, metadata={'help': 'Number of input channels.'}
    )
    out_channels: int = field(
        default=1, metadata={'help': 'Number of output channels.'}
    )
    features: int = field(
        default=64, metadata={'help': 'Number of features in first layer.'}
    )
    # Training
    batch_size: int = field(
        default=8, metadata={'help': 'Batch size for training.'}
    )
    learning_rate: float = field(
        default=1e-4, metadata={'help': 'Learning rate for training.'}
    )
    weight_decay: float = field(
        default=1e-4, metadata={'help': 'Weight decay for training.'}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={'help': 'Epsilon for Adam optimizer.'}
    )
