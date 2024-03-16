"""Module for whisper voice activity detection (VAD) using the WhisperX"""
import whisperx

if __name__ == '__main__':
    device = 'cuda'
    audio_file = 'audio.mp3'
    batch_size = 16  # reduce if low on GPU mem
    compute_type = 'float16'  # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model('large-v2', device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print(result['segments'])  # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device)
    result = whisperx.align(
        result['segments'], model_a, metadata, audio, device, return_char_alignments=False
    )

    print(result['segments'])  # after alignment
