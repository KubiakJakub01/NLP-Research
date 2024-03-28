import argparse
from pathlib import Path

import torchaudio
from tqdm import tqdm

from ..utils import normalize_audio


def get_params():
    parser = argparse.ArgumentParser(description='Normalize audio')
    parser.add_argument('--input_dir', type=Path, help='Input directory')
    parser.add_argument('--output_dir', type=Path, help='Output directory')
    parser.add_argument('--input_ext', type=str, default='.wav', help='Input extension')
    parser.add_argument('--output_ext', type=str, default='.wav', help='Output extension')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    return parser.parse_args()


def main(input_dir: Path, output_dir: Path, input_ext: str, output_ext: str, sample_rate: int):
    audio_fps = list(input_dir.glob(f'*{input_ext}'))
    for audio_fp in tqdm(audio_fps, desc='Normalizing audio'):
        output_fp = (output_dir / audio_fp.stem).with_suffix(output_ext)

        audio, orginal_sr = torchaudio.load(audio_fp)
        audio = normalize_audio(audio, orginal_sr, sample_rate)

        torchaudio.save(output_fp, audio, sample_rate)


if __name__ == '__main__':
    args = get_params()
    main(**vars(args))
