"""Module for whisper voice activity detection (VAD) using the WhisperX"""
import argparse
import json
from pathlib import Path

import whisperx
from tqdm import tqdm

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
        '--input_dir', '-i', required=True, type=Path, help='Path to dir with audios'
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        type=Path,
        default=None,
        help='Path to output dir. If `None` then it will print the output to stdout',
    )
    parser.add_argument(
        '--output_type',
        type=str,
        default='stdout',
        choices=['json', 'stdout'],
        help='Output type',
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
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=AVALIABLE_DTYPES,
        help='Data type for inference',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for inference. Reduce if low on GPU mem',
    )

    def _valid_params(params):
        if params.output_type == 'json' and params.output_dir is None:
            parser.error('Cannot use `json` as output dir is not provided')
        return params

    return _valid_params(parser.parse_args())


def main(
    input_dir: Path,
    output_dir: Path,
    output_type: str,
    audio_ext: str,
    model_size: str,
    lang: str,
    device: str,
    dtype: str,
    batch_size: int,
):
    model = whisperx.load_model(model_size, device, compute_type=dtype)
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
    audio_fp_list = list(input_dir.glob(f'*{audio_ext}'))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(audio_fp_list)) as pbar:
        for audio_fp in audio_fp_list:
            audio = whisperx.load_audio(audio_fp)
            result = model.transcribe(audio, batch_size=batch_size, language=lang)
            result = whisperx.align(
                result['segments'],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            if output_type == 'json':
                pbar.set_postfix_str(f'Processing {audio_fp.stem}')
                output_fp = output_dir / f'{audio_fp.stem}.json'
                with open(output_fp, 'w', encoding='utf-8') as f:
                    json.dump(result['segments'], f, indent=4)
            elif output_type == 'stdout':
                log_info('Processing %s', audio_fp.stem)
                log_info('Aligned result: %s', json.dumps(result['segments'], indent=4))


if __name__ == '__main__':
    main(**(vars(get_params())))
