import argparse
from pathlib import Path

from faster_whisper import WhisperModel

from ..utils import log_info

AVALIABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3']
AVALIABLE_DTYPES = [
    'int8',
    'int8_float32',
    'int8_float16',
    'int8_bfloat16',
    'int16',
    'float16',
    'bfloat16',
    'float32',
]


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-a', required=True, type=Path, help='Path to dir with audios'
    )
    parser.add_argument(
        '--output_fp',
        '-o',
        type=Path,
        default=None,
        help='Path to output file. \
            If path is `None` then output will be printed to stdout. \
            If path extension is `.json` then output will be saved in json format with timestamps. \
            If path extension is `.tsv` then output will be saved \
            in `tsv` format without timestamps.',
    )
    parser.add_argument('--audio_ext', '-e', type=str, default='.wav', help='Audio extension')
    parser.add_argument(
        '--model_size',
        '-s',
        type=str,
        default='base',
        choices=AVALIABLE_MODELS,
        help='Whisper model size',
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use'
    )
    parser.add_argument(
        '--dtype', type=str, default='float16', choices=AVALIABLE_DTYPES, help='Data type to use'
    )
    parser.add_argument('--beam_size', type=int, default=5, help='Number of beams during inference')
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()

    model = WhisperModel(params.model_size, device=params.device, compute_type=params.dtype)

    log_info('Load model with size: %s', params.model_size)

    segments, info = model.transcribe(params.audio_fp, beam_size=params.beam_size)

    log_info(
        'Detected language %s with probability %.2f', {info.language}, {info.language_probability}
    )

    for segment in segments:
        log_info('[%.2fs -> %.2fs] %s', segment.start, segment.end, segment.text)
