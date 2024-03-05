import argparse

from faster_whisper import WhisperModel


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_fp', '-a', type=str, help='Path to audio')
    parser.add_argument(
        '--model_size',
        '-s',
        type=str,
        choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'],
    )
    return parser.parse_args()


if __name__ == '__main__':
    model_size = 'base'

    # Run on GPU with FP16
    model = WhisperModel(model_size, device='cuda', compute_type='float16')

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe('audio.mp3', beam_size=5)

    print(f'Detected language {info.language} with probability {info.language_probability}')

    for segment in segments:
        print(f'[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}')
