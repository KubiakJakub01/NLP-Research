"""Sandbox for torchaudio squim implementation.

Based on the tutorial:
https://pytorch.org/audio/main/tutorials/squim_tutorial.html
"""

import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as F
from pesq import pesq
from pystoi import stoi
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
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


def predict_mos(waveform, title, sample_rate=16000):
    wav_numpy = waveform.numpy()

    sample_size = waveform.shape[1]
    time_axis = torch.arange(0, sample_size) / sample_rate

    figure, axes = plt.subplots(2, 1)
    axes[0].plot(time_axis, wav_numpy[0], linewidth=1)
    axes[0].grid(True)
    axes[1].specgram(wav_numpy[0], Fs=sample_rate)
    figure.suptitle(title)

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    mos = subjective_model(waveform)
    print(f'Predicted MOS for {title} is {mos[0]}')


def plot(waveform, title, sample_rate=16000):
    wav_numpy = waveform.numpy()

    sample_size = waveform.shape[1]
    time_axis = torch.arange(0, sample_size) / sample_rate

    figure, axes = plt.subplots(2, 1)
    axes[0].plot(time_axis, wav_numpy[0], linewidth=1)
    axes[0].grid(True)
    axes[1].specgram(wav_numpy[0], Fs=sample_rate)
    figure.suptitle(title)


def main():
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

    # Mix speech and noise
    snr_dbs = torch.tensor([20, -5])
    WAVEFORM_DISTORTED = F.add_noise(WAVEFORM_SPEECH, WAVEFORM_NOISE, snr_dbs)

    # Predict Objective metrics
    objective_model = SQUIM_OBJECTIVE.get_model()

    # Compare model output with ground truth for distorted speech with 20dB SNR
    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED[0:1, :])
    print(f'Estimated metrics for distorted speech at {snr_dbs[0]}dB are\n')
    print(f'STOI: {stoi_hyp[0]}')
    print(f'PESQ: {pesq_hyp[0]}')
    print(f'SI-SDR: {si_sdr_hyp[0]}\n')

    pesq_ref = pesq(16000, WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), mode='wb')
    stoi_ref = stoi(
        WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), 16000, extended=False
    )
    si_sdr_ref = si_snr(WAVEFORM_DISTORTED[0:1], WAVEFORM_SPEECH)
    print(f'Reference metrics for distorted speech at {snr_dbs[0]}dB are\n')
    print(f'STOI: {stoi_ref}')
    print(f'PESQ: {pesq_ref}')
    print(f'SI-SDR: {si_sdr_ref}')

    # Compare model output with ground truth for distorted speech with -5dB SNR
    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED[1:2, :])
    print(f'Estimated metrics for distorted speech at {snr_dbs[1]}dB are\n')
    print(f'STOI: {stoi_hyp[0]}')
    print(f'PESQ: {pesq_hyp[0]}')
    print(f'SI-SDR: {si_sdr_hyp[0]}\n')

    pesq_ref = pesq(16000, WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[1].numpy(), mode='wb')
    stoi_ref = stoi(
        WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[1].numpy(), 16000, extended=False
    )
    si_sdr_ref = si_snr(WAVEFORM_DISTORTED[1:2], WAVEFORM_SPEECH)
    print(f'Reference metrics for distorted speech at {snr_dbs[1]}dB are\n')
    print(f'STOI: {stoi_ref}')
    print(f'PESQ: {pesq_ref}')
    print(f'SI-SDR: {si_sdr_ref}')


if __name__ == '__main__':
    main()
