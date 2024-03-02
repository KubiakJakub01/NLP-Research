import argparse
from pathlib import Path

import sox
from tqdm import tqdm


def get_params():
    parser = argparse.ArgumentParser(description='Normalize audio')
    parser.add_argument('--input_dir', type=Path, help='Input directory')
    parser.add_argument('--output_dir', type=Path, help='Output directory')
    parser.add_argument('--input_ext', type=str, default='.wav', help='Input extension')
    parser.add_argument('--output_ext', type=str, default='.wav', help='Output extension')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate')
    return parser.parse_args()


def normalize_audio(input_dir, output_dir, input_ext='.wav', output_ext='.wav', sample_rate=16000):
    input_files = list(input_dir.glob(f'*{input_ext}'))
    output_dir.mkdir(parents=True, exist_ok=True)

    tfm = sox.Transformer()
    tfm.set_output_format(file_type=input_ext.split('.')[-1], rate=sample_rate)

    for input_file in tqdm(input_files, desc='Normalizing audio'):
        output_file = output_dir / (input_file.stem + output_ext)
        tfm.build(input_file, output_file)


def main():
    args = get_params()
    normalize_audio(
        args.input_dir, args.output_dir, args.input_ext, args.output_ext, args.sample_rate
    )


if __name__ == '__main__':
    main()
