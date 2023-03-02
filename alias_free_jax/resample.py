import flax.linen as nn
import jax.numpy as jnp

from .filter import kaiser_sinc_filter1d, LowPassFilter1d
from .lax import conv_transpose


class UpSample1d(nn.Module):
    ratio: int = 2
    kernel_size: int = 12

    def setup(self):
        self.stride = self.ratio
        self.pad = self.kernel_size // self.ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )

        filter_ = kaiser_sinc_filter1d(
            cutoff=0.5 / self.ratio,
            half_width=0.6 / self.ratio,
            kernel_size=self.kernel_size,
        )
        self.filter = self.variable("params", "filter", lambda: filter_)

    def __call__(self, x):
        _, _, C = x.shape

        x = jnp.pad(x, ((0, 0), (self.pad, self.pad), (0, 0)), mode="edge")
        x = self.ratio * conv_transpose(
            x,
            jnp.repeat(self.filter.value, C, axis=2),
            strides=(self.ratio,),
            padding="VALID",
            feature_group_count=C,
        )
        x = x[..., self.pad_left : -self.pad_right, :]
        return x


class DownSample1d(nn.Module):
    ratio: int = 2
    kernel_size: int = 12

    def setup(self):
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / self.ratio,
            half_width=0.6 / self.ratio,
            stride=self.ratio,
            kernel_size=self.kernel_size,
        )

    def __call__(self, x):
        xx = self.lowpass(x)

        return xx
