import argparse
from pathlib import Path

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
    normalize_audio(input_dir, output_dir, input_ext, output_ext, sample_rate)


if __name__ == '__main__':
    args = get_params()
    main(**vars(args))
