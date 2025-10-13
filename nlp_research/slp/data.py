"""Module with pytorch datasets and dataloaders for SLP tasks."""

from pathlib import Path

import torchaudio
from einops import rearrange
from torch.utils.data import Dataset

from ..utils import normalize_audio


class AudioDataset(Dataset):
    def __init__(self, data_dir: Path, audio_ext: str = '.wav', target_sr: int = 16000):
        self.data_dir = data_dir
        self.audio_ext = audio_ext
        self.target_sr = target_sr
        self.audio_fps = list(data_dir.glob(f'*{audio_ext}'))

    def __len__(self):
        return len(self.audio_fps)

    def __getitem__(self, idx):
        audio_fp = self.audio_fps[idx]
        audio, sr = torchaudio.load(audio_fp)
        audio = rearrange(normalize_audio(audio, sr, self.target_sr), '1 t -> t')
        return audio.numpy()
