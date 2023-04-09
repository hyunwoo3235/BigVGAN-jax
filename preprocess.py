import argparse
import glob
import os
import random

import librosa
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import read
from tqdm.auto import tqdm

from utils import convert_audio
from meldataset import mel_spectrogram


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_name", default="kss")
    parser.add_argument(
        "--files_dir",
        default="data",
    )
    parser.add_argument("--add_blank", default=True, type=bool)
    parser.add_argument("--segment_size", default=8192, type=int)
    parser.add_argument("--sampling_rate", default=24000, type=int)
    parser.add_argument("--num_mels", default=100, type=int)
    parser.add_argument("--n_fft", default=1024, type=int)
    parser.add_argument("--hop_size", default=256, type=int)
    parser.add_argument("--win_size", default=1024, type=int)

    args = parser.parse_args()

    filepaths = glob.glob(f"{args.files_dir}/**/*.wav", recursive=True)

    with tf.io.TFRecordWriter(
        f"{args.out_name}.tfrecord",
        options=tf.io.TFRecordOptions(compression_type="GZIP"),
    ) as writer:
        for filepath in tqdm(filepaths):
            if not os.path.exists(filepath):
                continue

            sr, audio = read(filepath)
            audio = convert_audio(audio)
            audio = librosa.to_mono(audio.T.astype(np.float32))

            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sampling_rate)

            if audio.shape[0] > args.segment_size:
                max_audio_start = audio.shape[0] - args.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start : audio_start + args.segment_size]
            else:
                audio = np.pad(
                    audio, (0, max(0, args.segment_size - audio.shape[0])), "constant"
                )

            mel = mel_spectrogram(
                audio.reshape(1, -1, 1),
                n_fft=args.n_fft,
                num_mels=args.num_mels,
                sampling_rate=args.sampling_rate,
                hop_size=args.hop_size,
                win_size=args.win_size,
                fmin=0,
                fmax=None,
            )

            audio = audio.astype(np.float32)
            mel = mel.astype(np.float32)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "mel": _bytes_feature(mel.tobytes()),
                        "audio": _bytes_feature(audio.tobytes()),
                    }
                )
            )
            writer.write(example.SerializeToString())
