import flax.linen as nn

from .resample import UpSample1d, DownSample1d


class Activation1d(nn.Module):
    act: nn.Module = nn.relu
    up_ratio: int = 2
    down_ratio: int = 2
    up_kernel_size: int = 12
    down_kernel_size: int = 12

    @nn.compact
    def __call__(self, x):
        x = UpSample1d(
            ratio=self.up_ratio,
            kernel_size=self.up_kernel_size,
            name="upsample",
        )(x)
        x = self.act(x)
        x = DownSample1d(
            ratio=self.down_ratio,
            kernel_size=self.down_kernel_size,
            name="downsample",
        )(x)

        return x
