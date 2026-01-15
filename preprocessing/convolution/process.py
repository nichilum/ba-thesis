import numpy as np
from scipy.signal import convolve
from scipy.signal import resample_poly


def resample_stereo(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample stereo float array shaped (2, N) from sr_in to sr_out."""
    if sr_in == sr_out:
        return x
    if sr_in <= 0 or sr_out <= 0:
        raise ValueError(f"Invalid sample rates: sr_in={sr_in}, sr_out={sr_out}")

    # Use rational resampling with integer factors.
    from math import gcd

    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g

    # resample_poly supports an axis parameter.
    return resample_poly(x, up=up, down=down, axis=1).astype(np.float64, copy=False)


def wav_to_float_stereo(x: np.ndarray) -> np.ndarray:
    """Convert wavfile.read() output to float64 stereo array shaped (2, N).

    - If mono, the channel is duplicated.
    - If more than 2 channels, the first two are used.
    - Values are scaled to [-1, 1] for integer PCM types.
    """
    x = np.asarray(x)

    # Normalize shape to (N, C)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"Unsupported WAV array shape: {x.shape}")

    # Ensure 2 channels
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]

    # Convert to float64 and scale
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float64) / max(abs(info.min), info.max)
    else:
        # float WAVs are typically already in [-1, 1]
        y = x.astype(np.float64)

    # Return as (2, N)
    return y.T


def apply_convolution_reverb(
    sample: np.ndarray,
    samplerate_sample: int,
    reverb: np.ndarray,
    samplerate_reverb: int,
    master_samplerate: int,
) -> np.ndarray:
    sample = wav_to_float_stereo(sample)
    num_samples_sample = sample.shape[1]
    num_channels_sample = sample.shape[0]

    if samplerate_sample != master_samplerate:
        sample = resample_stereo(sample, samplerate_sample, master_samplerate)
        samplerate_sample = master_samplerate
        num_samples_sample = sample.shape[1]

    reverb = wav_to_float_stereo(reverb)

    if samplerate_reverb != master_samplerate:
        reverb = resample_stereo(reverb, samplerate_reverb, master_samplerate)
        samplerate_reverb = master_samplerate

    num_samples_reverb = reverb.shape[1]
    num_channels_reverb = reverb.shape[0]

    if num_channels_reverb > num_channels_sample:
        # remove extra channels from reverb
        reverb = reverb[:num_channels_sample, :]
        num_channels_reverb = num_channels_sample
    elif num_channels_reverb < num_channels_sample:
        # pad missing channels by duplicating the first channel
        reverb = np.vstack([reverb, reverb[:1, :]])
        num_channels_reverb = num_channels_sample

    total_samples_reverb = num_samples_reverb * num_channels_reverb

    sample_max = np.maximum(np.max(np.abs(sample), axis=1, keepdims=True), 1e-12)
    sample = sample / sample_max

    reverb_max = np.maximum(np.max(np.abs(reverb), axis=1, keepdims=True), 1e-12)
    reverb = reverb / reverb_max

    #   MAIN PART OF THE ALGORITHM   #

    gain_dry = 1
    gain_wet = 1
    output_gain = 0.05

    reverb_out = np.zeros(
        [2, np.shape(sample)[1] + np.shape(reverb)[1] - 1], dtype=np.float64
    )
    reverb_out[0] = output_gain * (
        convolve(sample[0] * gain_dry, reverb[0] * gain_wet, method="fft")
    )
    reverb_out[1] = output_gain * (
        convolve(sample[1] * gain_dry, reverb[1] * gain_wet, method="fft")
    )

    #   WRITING TO FILE   #

    reverb_integer = np.zeros((reverb_out.shape))

    reverb_integer[0] = (reverb_out[0] * int(np.iinfo(np.int16).max)).astype(np.int16)
    reverb_integer[1] = (reverb_out[1] * int(np.iinfo(np.int16).max)).astype(np.int16)

    reverb_to_render = np.empty(
        (reverb_integer[0].size + reverb_integer[1].size), dtype=np.int16
    )
    reverb_to_render[0::2] = reverb_integer[0]
    reverb_to_render[1::2] = reverb_integer[1]

    return reverb_to_render
