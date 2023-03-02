import math

import flax.linen as nn
import jax.numpy as jnp
from jax import lax

from .lax import conv_dimension_numbers


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A > 21.0:
        beta = 0.5842 * (A - 21.0) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = jnp.kaiser(kernel_size, beta)

    if even:
        time = jnp.arange(-half_size, half_size) + 0.5
    else:
        time = jnp.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = jnp.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * jnp.sinc(2 * cutoff * time)
        filter_ /= jnp.sum(filter_)
        filter_ = jnp.reshape(filter_, (kernel_size, 1, 1))

    return filter_


class LowPassFilter1d(nn.Module):
    cutoff: float = 0.5
    half_width: float = 0.6
    stride: int = 1
    padding: bool = True
    padding_mode: str = "edge"
    kernel_size: int = 12

    def setup(self):
        self.even = self.kernel_size % 2 == 0
        self.pad_left = self.kernel_size // 2 - int(self.even)
        self.pad_right = self.kernel_size // 2

        filter_ = kaiser_sinc_filter1d(self.cutoff, self.half_width, self.kernel_size)
        self.filter = self.variable("params", "filter", lambda: filter_)

    @nn.compact
    def __call__(self, x):
        _, _, C = x.shape

        if self.padding:
            x = jnp.pad(
                x,
                ((0, 0), (self.pad_left, self.pad_right), (0, 0)),
                mode=self.padding_mode,
            )
        dn = conv_dimension_numbers(x.shape)
        out = lax.conv_general_dilated(
            x,
            jnp.repeat(self.filter.value, C, axis=2),
            window_strides=(self.stride,),
            padding="VALID",
            dimension_numbers=dn,
            feature_group_count=C,
        )

        return out
