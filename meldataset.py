import glob
import os
from typing import Optional

import jax.numpy as jnp
import tensorflow as tf
from audax.core.functional import melscale_fbanks, spectrogram, apply_melscale

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
    return jnp.log(jnp.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression_jax(x, C=1):
    return jnp.exp(x) / C


def spectral_normalize_jax(magnitudes):
    output = dynamic_range_compression_jax(magnitudes)
    return output


def spectral_de_normalize_jax(magnitudes):
    output = dynamic_range_decompression_jax(magnitudes)
    return output


mel_basis = None
hann_window = None


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    global hann_window, mel_basis
    if hann_window is None:
        hann_window = jnp.hanning(win_size)
    if mel_basis is None:
        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            n_mels=num_mels,
            sample_rate=sampling_rate,
            f_min=fmin,
            f_max=fmax,
            norm="slaney",
            mel_scale="slaney",
        )

    y = jnp.pad(
        y,
        ((0, 0), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), (0, 0)),
        mode="reflect",
    )

    spec = spectrogram(
        y,
        pad=0,
        window=hann_window,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        power=1.0,
        normalized=False,
        center=center,
        onesided=True,
    )

    mel = apply_melscale(spec, mel_basis)
    mel = spectral_normalize_jax(mel)
    return mel


def get_dataset(
    dataset_path,
    batch_size,
    num_mels: int = 100,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    compression_type: Optional[str] = "GZIP",
):
    def _parse_function(example_proto):
        feature_description = {
            "mel": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        audio = tf.io.decode_raw(parsed_features["audio"], tf.float32)
        mel = tf.io.decode_raw(parsed_features["mel"], tf.float32)
        mel = tf.reshape(mel, (-1, num_mels))
        return mel, audio

    if os.path.isdir(dataset_path):
        tfrecords = glob.glob(os.path.join(dataset_path, "*.tfrecord"))
    else:
        tfrecords = [dataset_path]

    dataset = (
        tf.data.TFRecordDataset(tfrecords, compression_type=compression_type)
        .map(_parse_function)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
