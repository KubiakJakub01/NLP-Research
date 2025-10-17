"""Utility functions for NLP research."""

import logging
import random

import coloredlogs
import numpy as np
import torch
import torchaudio.functional as F

# Set up logging
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('debug.log', mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
file_handler.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(file_handler)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')


def log_debug(*args, **kwargs):
    """Log an debug message."""
    logger.debug(*args, **kwargs)


def log_info(*args, **kwargs):
    """Log an info message."""
    logger.info(*args, **kwargs)


def log_warning(*args, **kwargs):
    """Log a warning message."""
    logger.warning(*args, **kwargs)


def log_error(*args, **kwargs):
    """Log an error message."""
    logger.error(*args, **kwargs)


def seed(seed_number: int):
    """Set the random seed for reproducibility."""
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True


def normalize_audio(audio: torch.Tensor, orginal_sr: int, target_sr: int) -> torch.Tensor:
    """Normalize audio to momo channel and target sample rate."""
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if orginal_sr != target_sr:
        audio = F.resample(audio, orginal_sr, target_sr)
    return audio
