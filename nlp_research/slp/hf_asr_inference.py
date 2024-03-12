"""Script with the ASR inference using Hugging Face's model."""
import argparse
from pathlib import Path

from transformers import pipeline

from ..utils import log_info

AVALIABLE_MODELS = [
    'openai/whisper-large-v3',
    'openai/whisper-large-v2',
    'openai/whisper-medium',
    'openai/whisper-small',
    'openai/whisper-base',
    'openai/whisper-tiny',
]


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-i', required=True, type=Path, help='Path to dir with audios'
    )
    parser.add_argument(
        '--output_fp',
        '-o',
        type=Path,
        help='Path to output `.tsv` file',
    )
    parser.add_argument('--audio_ext', '-e', type=str, default='.wav', help='Audio extension')
    parser.add_argument(
        '--model_size',
        '-s',
        type=str,
        default='openai/whisper-base',
        choices=AVALIABLE_MODELS,
        help='Whisper model size',
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        default=1,
        help='Batch size for inference',
    )
    parser.add_argument(
        '--lang',
        '-l',
        type=str,
        default=None,
        help='Language to use for inference. \
            If `None` then language will be detected automatically',
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use'
    )
    return parser.parse_args()


def main(
    input_dir: Path,
    output_fp: Path,
    audio_ext: str,
    model_size: str,
    lang: str,
    batch_size: int,
    device: str,
):
    # Load the model
    pipe = pipeline(
        'automatic-speech-recognition',
        model=model_size,
        device=device,
        chunk_length_s=30,
        generate_kwargs={'language': lang},
    )

    audio_fp_list = list(input_dir.glob(f'*{audio_ext}'))
    log_info(f'Found {len(audio_fp_list)} files')

    output_fp.parent.mkdir(exist_ok=True, parents=True)

    for audio_fp in audio_fp_list:
        transcription = pipe(audio_fp.as_posix(), batch_size=batch_size)['text']
        log_info(f'Processed {audio_fp}')
        log_info(f'{transcription=}')


if __name__ == '__main__':
    args = get_params()
    main(**vars(args))
