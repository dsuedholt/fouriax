#!/usr/bin/env python

"""Tests for `fouriax` package."""

import jax
import numpy as np
import torch
import math
from auraloss.freq import MultiResolutionSTFTLoss, STFTLoss, SumAndDifferenceSTFTLoss, MelSTFTLoss
from hypothesis import given, settings
from hypothesis import strategies as st

import librosa

from fouriax.freq import multi_resolution_stft_loss, stft_loss, sum_and_difference_stft_loss

multi_resolution_stft_loss, stft_loss = jax.jit(
    multi_resolution_stft_loss,
    static_argnames=[
        "window",
        "fft_sizes",
        "hop_sizes",
        "win_lengths",
        "w_sc",
        "w_log_mag",
        "w_lin_mag",
        "w_phs",
        "perceptual_weighting",
        "scale_invariance",
        "eps",
        "output",
        "reduction",
        "mag_distance",
    ],
), jax.jit(
    stft_loss,
    static_argnames=[
        "window",
        "fft_size",
        "hop_size",
        "win_length",
        "w_sc",
        "w_log_mag",
        "w_lin_mag",
        "w_phs",
        "perceptual_weighting",
        "scale_invariance",
        "eps",
        "output",
        "reduction",
        "mag_distance",
    ],
)

shared_shape = st.shared(
    st.tuples(
        st.integers(min_value=1, max_value=16),
        st.integers(min_value=2048, max_value=4096),
        st.just(1),
    ),
    key="array_shape",
)


fs = 44100  # Sampling rate


@st.composite
def generate_sine_wave(draw, length):
    """Composite strategy to generate a single sine wave."""
    amplitude = draw(st.floats(min_value=0.01, max_value=1.0))
    frequency = draw(st.floats(min_value=30.0, max_value=22050.0))
    t = np.linspace(0, length / fs, int(length), endpoint=False, dtype=np.float32)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return sine_wave


@st.composite
def generate_complex_signal(draw, shape):
    """Composite strategy to generate a complex signal from multiple sine waves."""
    length = math.prod(shape)  # Using the second element of shape for signal length
    sine_waves = draw(st.lists(generate_sine_wave(length), min_size=32, max_size=32))
    complex_signal = np.sum(sine_waves, axis=0)
    # Normalize the complex signal to ensure it's within [-1, 1]
    max_amplitude = np.max(np.abs(complex_signal))
    if max_amplitude > 0:  # Avoid division by zero
        complex_signal /= max_amplitude
    return complex_signal.reshape(shape)


# Define the audio strategy
audio_strategy = generate_complex_signal((1, fs, 1))
stereo_strategy = generate_complex_signal((1, fs, 2))

@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
    st.integers(min_value=6, max_value=8).map(lambda x: 2**x),
)
def test_stft_loss(inputs, target, res):
    """Sample pytest test function with the pytest fixture as an argument."""
    loss = stft_loss(inputs, target, res, res // 4, res // 2)
    loss_ref = STFTLoss(res, res // 4, res // 2)(
        torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref, atol=1.0e-1)


@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
    st.integers(min_value=6, max_value=8).map(lambda x: 2**x),
)
def test_scale_invariant_loss(inputs, target, res):
    loss = stft_loss(inputs, target, res, res // 4, res // 2, scale_invariance=True)
    loss_ref = STFTLoss(res, res // 4, res // 2, scale_invariance=True)(
        torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref, atol=1.0e-1)

@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
)
def test_multi_resolution_stft_loss(inputs, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    fft_sizes = (256, 512)
    hop_sizes = (64, 128)
    win_lengths = (128, 256)

    loss = multi_resolution_stft_loss(
        inputs, target, fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths
    )
    loss_ref = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths
    )(
        torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref, atol=1.0e-1)


@settings(deadline=None, max_examples=10)
@given(
    audio_strategy,
    audio_strategy,
    st.integers(min_value=10, max_value=12).map(lambda x: 2**x),
    st.integers(min_value=5, max_value=7).map(lambda x: 2**x),
)
def test_mel_scale_stft_loss(inputs, target, res, n_mels):
    """Sample pytest test function with the pytest fixture as an argument."""
    mel_fb = librosa.filters.mel(sr=fs, n_fft=res, n_mels=n_mels)
    loss = stft_loss(inputs, target, res, res // 4, res // 2, scale=mel_fb)
    loss_ref = MelSTFTLoss(fs, res, res // 4, res // 2, n_mels=n_mels)(
        torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref, atol=1.0e-1)

@settings(deadline=None, max_examples=10)
@given(
    stereo_strategy,
    stereo_strategy,
)
def test_sum_and_difference_stft_loss(inputs, target):
    """Sample pytest test function with the pytest fixture as an argument."""
    fft_sizes = (256, 512)
    hop_sizes = (64, 128)
    win_lengths = (128, 256)

    loss = sum_and_difference_stft_loss(
        inputs,
        target,
        fft_sizes=fft_sizes,
        hop_sizes=hop_sizes,
        win_lengths=win_lengths,
        output="loss",
        ch_axis=2,
    )
    loss_ref = SumAndDifferenceSTFTLoss(
        fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths
    )(
        torch.from_numpy(np.transpose(inputs, (0, 2, 1))),
        torch.from_numpy(np.transpose(target, (0, 2, 1))),
    )
    assert np.allclose(loss, loss_ref, atol=1.0e-1)