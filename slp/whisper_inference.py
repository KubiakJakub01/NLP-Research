import argparse

from faster_whisper import WhisperModel


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', '-a', required=True, type=str, help='Path to dir with audios'
    )
    parser.add_argument('--audio_ext', '-e', type=str, default='.wav', help='Audio extension')
    parser.add_argument(
        '--model_size',
        '-s',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'],
        help='Whisper model size',
    )
    parser.add_argument(
        '--device', '-d', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use'
    )
    parser.add_argument('--dtype', type=str, default='float16', help='Data type to use')
    parser.add_argument('--beam_size', type=int, default=5, help='Number of beams during inference')
    return parser.parse_args()


if __name__ == '__main__':
    params = get_params()

    # Run on GPU with FP16
    model = WhisperModel(params.model_size, device=params.device, compute_type=params.dtype)

    print(f'Load model with size: {params.model_size}')

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(params.audio_fp, beam_size=params.beam_size)

    print(f'Detected language {info.language} with probability {info.language_probability}')

    for segment in segments:
        print(f'[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}')
