"""Module with utility functions for SLP tasks."""
import torch
import torchaudio.functional as F


def normalize_audio(audio: torch.Tensor, orginal_sr: int, target_sr: int) -> torch.Tensor:
    """Normalize audio to momo channel and target sample rate."""
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if orginal_sr != target_sr:
        audio = F.resample(audio, orginal_sr, target_sr)
    return audio
