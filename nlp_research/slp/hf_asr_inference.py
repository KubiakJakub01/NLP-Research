"""Script with the ASR inference using Hugging Face's model."""
import argparse
from pathlib import Path

from transformers import pipeline

from ..utils import log_info
from .data import AudioDataset

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
        default=None,
        type=Path,
        help='Path to output `.tsv` file. If `None` then output will be printed to the console.',
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
    output_fp: Path | None,
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

    # Load the dataset
    audio_dataset = AudioDataset(input_dir, audio_ext)
    log_info('Found %d audio files', len(audio_dataset))

    # Inference
    if output_fp:
        output_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(output_fp, 'w') as fh:
            fh.write('AudioName\tTranscription\n')
    for audio_fp, transcription in zip(
        audio_dataset.audio_fps, pipe(audio_dataset, batch_size=batch_size), strict=False
    ):
        transcription = transcription['text']
        if output_fp:
            with open(output_fp, 'a') as fh:
                fh.write(f'{audio_fp.name}\t{transcription}\n')
        else:
            log_info('%s\t%s', audio_fp.name, transcription)


if __name__ == '__main__':
    args = get_params()
    main(**vars(args))
