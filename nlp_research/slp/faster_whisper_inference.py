import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm

from ..utils import log_debug, log_info

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
        default='json',
        choices=['json', 'tsv', 'stdout'],
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
        '--dtype', type=str, default='float16', choices=AVALIABLE_DTYPES, help='Data type to use'
    )
    parser.add_argument('--beam_size', type=int, default=5, help='Number of beams during inference')
    return parser.parse_args()


def segments_to_json(segments):
    json_element_list = []
    for segment in segments:
        json_element_list.append(
            {
                'id': segment.id,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
            }
        )
    return json_element_list


def segments_to_tsv(segments):
    return ' '.join([segment.text for segment in segments])


def main(
    model_size: str,
    device: str,
    dtype: str,
    input_dir: Path,
    output_dir: Path,
    output_type: str,
    lang: str,
    audio_ext: str,
    beam_size: int,
):
    model = WhisperModel(model_size, device=device, compute_type=dtype)
    log_info('Load model with size: %s', model_size)

    audio_fps = list(input_dir.glob(f'*{audio_ext}'))
    log_info('Found %s audio files', len(audio_fps))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_tsv_fp = output_dir / 'transcriptions.tsv'
        if output_type == 'tsv':
            output_tsv_fp.write_text('FileID\tTranscription\n')

    with tqdm(total=len(audio_fps), desc='Inference') as pbar:
        for audio_fp in audio_fps:
            segments, info = model.transcribe(
                audio_fp.as_posix(), beam_size=beam_size, language=lang
            )

            log_debug(
                'Detected language %s with probability %.2f',
                info.language,
                info.language_probability,
            )

            if output_type == 'stdout':
                log_info('File: %s', audio_fp)
                for segment in segments:
                    log_info(
                        '%d [%.2fs -> %.2fs] %s',
                        segment.id,
                        segment.start,
                        segment.end,
                        segment.text,
                    )
            elif output_type == 'json':
                json_output_fp = output_dir / audio_fp.with_suffix('.json').name
                with json_output_fp.open('w', encoding='utf-8') as f:
                    json.dump(
                        {
                            'audio_fp': audio_fp.as_posix(),
                            'segments': segments_to_json(segments),
                        },
                        f,
                        indent=4,
                    )
                pbar.update(1)
            elif output_type == 'tsv':
                with output_tsv_fp.open('a', encoding='utf-8') as f:
                    f.write(f'{audio_fp.stem}\t{segments_to_tsv(segments)}\n')
                pbar.update(1)


if __name__ == '__main__':
    params = get_params()
    main(**(vars(params)))
