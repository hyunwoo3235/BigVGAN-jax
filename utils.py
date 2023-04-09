import numpy as np
from scipy.io.wavfile import read


def load_wav(full_path, sr_target):
    sampling_rate, data = read(full_path)
    if sampling_rate != sr_target:
        raise RuntimeError(
            "Sampling rate of the file {} is {} Hz, but the model requires {} Hz".format(
                full_path, sampling_rate, sr_target
            )
        )
    return data, sampling_rate


def convert_audio(wav):
    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    return wav


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
