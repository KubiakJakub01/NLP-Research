"""Sandbox for torchaudio squim implementation.

Based on the tutorial:
https://pytorch.org/audio/main/tutorials/squim_tutorial.html
"""

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.utils import download_asset


def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()


def plot(waveform, title, sample_rate=16000):
    wav_numpy = waveform.numpy()

    sample_size = waveform.shape[1]
    time_axis = torch.arange(0, sample_size) / sample_rate

    figure, axes = plt.subplots(2, 1)
    axes[0].plot(time_axis, wav_numpy[0], linewidth=1)
    axes[0].grid(True)
    axes[1].specgram(wav_numpy[0], Fs=sample_rate)
    figure.suptitle(title)


if __name__ == '__main__':
    # Download assets
    SAMPLE_SPEECH = download_asset(
        'tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
    )
    SAMPLE_NOISE = download_asset('tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav')

    # Load audio from file
    WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load(SAMPLE_SPEECH)
    WAVEFORM_NOISE, SAMPLE_RATE_NOISE = torchaudio.load(SAMPLE_NOISE)
    WAVEFORM_NOISE = WAVEFORM_NOISE[0:1, :]

    # Resample
    if SAMPLE_RATE_SPEECH != 16000:
        WAVEFORM_SPEECH = F.resample(WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH, 16000)

    if SAMPLE_RATE_NOISE != 16000:
        WAVEFORM_NOISE = F.resample(WAVEFORM_NOISE, SAMPLE_RATE_NOISE, 16000)

    # Make waveform the same length
    if WAVEFORM_SPEECH.shape[1] < WAVEFORM_NOISE.shape[1]:
        WAVEFORM_NOISE = WAVEFORM_NOISE[:, : WAVEFORM_SPEECH.shape[1]]
    else:
        WAVEFORM_SPEECH = WAVEFORM_SPEECH[:, : WAVEFORM_NOISE.shape[1]]
