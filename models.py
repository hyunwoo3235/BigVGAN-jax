from typing import Tuple, Sequence, Union, Optional, List, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from audax.core.functional import spectrogram

import activations
from alias_free_jax import Activation1d

LRELU_SLOPE = 0.1


class FlaxConvWithWeightNorm(nn.Module):
    in_features: int
    out_features: int
    kernel_size: Sequence[int]
    strides: int = 1
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "SAME"
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=self.out_features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=jax.nn.initializers.he_normal(),
            padding=self.padding,
            feature_group_count=self.feature_group_count,
            dtype=self.dtype,
        )
        weight_shape = self.kernel_size + (
            self.in_features // self.feature_group_count,
            self.out_features,
        )
        self.weight_v = self.param(
            "weight_v", jax.nn.initializers.he_normal(), weight_shape
        )
        self.weight_g = self.param(
            "weight_g",
            lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :],
        )
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.conv.features,))

    def _get_normed_weights(self):
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    def __call__(self, hidden_states):
        kernel = self._get_normed_weights()
        hidden_states = self.conv.apply(
            {"params": {"kernel": kernel, "bias": self.bias}}, hidden_states
        )
        return hidden_states


class FlaxConvTransposeWithWeightNorm(nn.Module):
    in_features: int
    out_features: int
    kernel_size: Sequence[int]
    strides: int = 1
    padding: Union[str, int, Sequence[Union[int, Tuple[int, int]]]] = "SAME"
    kernel_dilation: Optional[Sequence[int]] = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_transpose = nn.ConvTranspose(
            features=self.out_features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            kernel_init=jax.nn.initializers.he_normal(),
            padding=self.padding,
            kernel_dilation=self.kernel_dilation,
            dtype=self.dtype,
        )
        weight_shape = self.kernel_size + (
            self.in_features,
            self.out_features,
        )
        self.weight_v = self.param(
            "weight_v", jax.nn.initializers.he_normal(), weight_shape
        )
        self.weight_g = self.param(
            "weight_g",
            lambda _: jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :],
        )
        self.bias = self.param(
            "bias", jax.nn.initializers.zeros, (self.conv_transpose.features,)
        )

    def _get_normed_weights(self):
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    def __call__(self, hidden_states):
        kernel = self._get_normed_weights()
        hidden_states = self.conv_transpose.apply(
            {"params": {"kernel": kernel, "bias": self.bias}}, hidden_states
        )
        return hidden_states


class AMPBlock1(nn.Module):
    channels: int
    kernel_size: int = 3
    dilations: Tuple[int] = (1, 3, 5)
    activation: str = "snake"
    alpha_logscale: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.convs1 = [
            FlaxConvWithWeightNorm(
                self.channels,
                self.channels,
                (self.kernel_size,),
                kernel_dilation=dilation,
                dtype=self.dtype,
            )
            for dilation in self.dilations
        ]
        self.convs2 = [
            FlaxConvWithWeightNorm(
                self.channels,
                self.channels,
                (self.kernel_size,),
                kernel_dilation=1,
                dtype=self.dtype,
            )
            for _ in self.dilations
        ]

        self.num_layers = len(self.convs1) + len(self.convs2)

        if self.activation == "snake":
            self.activations = [
                Activation1d(
                    activations.Snake(
                        self.channels, alpha_logscale=self.alpha_logscale
                    ),
                )
                for _ in range(self.num_layers)
            ]
        elif self.activation == "snakebeta":
            self.activations = [
                Activation1d(
                    activations.SnakeBeta(
                        self.channels, alpha_logscale=self.alpha_logscale
                    ),
                )
                for _ in range(self.num_layers)
            ]
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def __call__(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x


class BigVGAN(nn.Module):
    num_mels: int
    resblock_kernel_sizes: list
    resblock_dilation_sizes: list
    upsample_rates: tuple
    upsample_initial_channel: int
    upsample_kernel_sizes: list
    activation: str = "snake"
    alpha_logscale: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)

        self.conv_pre = FlaxConvWithWeightNorm(
            self.num_mels, self.upsample_initial_channel, (7,), (1,), dtype=self.dtype
        )

        ups = []
        for i, (u, k) in enumerate(
            zip(self.upsample_rates, self.upsample_kernel_sizes)
        ):
            ups.append(
                FlaxConvTransposeWithWeightNorm(
                    self.upsample_initial_channel // (2**i),
                    self.upsample_initial_channel // (2 ** (i + 1)),
                    (k,),
                    (u,),
                    dtype=self.dtype,
                )
            )
        self.ups = ups

        resblocks = []
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)
            ):
                resblocks.append(
                    AMPBlock1(
                        ch, k, d, self.activation, self.alpha_logscale, self.dtype
                    )
                )
        self.resblocks = resblocks

        if self.activation == "snake":
            self.activation_post = Activation1d(
                activations.Snake(ch, alpha_logscale=self.alpha_logscale)
            )
        elif self.activation == "snakebeta":
            self.activation_post = Activation1d(
                activations.SnakeBeta(ch, alpha_logscale=self.alpha_logscale)
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.conv_post = FlaxConvWithWeightNorm(ch, 1, (7,), (1,))

    def __call__(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = nn.tanh(x)

        return x


class DiscriminatorP(nn.Module):
    period: int
    d_mult: int = 1
    kernel_size: int = 5
    stride: int = 3
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.use_spectral_norm:
            raise NotImplementedError

        self.convs = [
            FlaxConvWithWeightNorm(
                1, 32 * self.d_mult, (self.kernel_size, 1), (self.stride, 1)
            ),
            FlaxConvWithWeightNorm(
                32 * self.d_mult,
                128 * self.d_mult,
                (self.kernel_size, 1),
                (self.stride, 1),
            ),
            FlaxConvWithWeightNorm(
                128 * self.d_mult,
                512 * self.d_mult,
                (self.kernel_size, 1),
                (self.stride, 1),
            ),
            FlaxConvWithWeightNorm(
                512 * self.d_mult,
                1024 * self.d_mult,
                (self.kernel_size, 1),
                (self.stride, 1),
            ),
            FlaxConvWithWeightNorm(
                1024 * self.d_mult, 1024 * self.d_mult, (self.kernel_size, 1), 1
            ),
        ]
        self.conv_post = FlaxConvWithWeightNorm(1024 * self.d_mult, 1, (3, 1), 1)

    def __call__(self, x):
        fmap = []

        b, t, c = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, ((0, 0), (0, n_pad), (0, 0)), mode="reflect")
            t = t + n_pad
        x = x.reshape(b, t // self.period, self.period, c)

        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.reshape(1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    mpd_reshapes: list = (2, 3, 5, 7, 11)
    d_mult: int = 1
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.use_spectral_norm:
            raise NotImplementedError
        self.discriminators = [
            DiscriminatorP(
                period, self.d_mult, use_spectral_norm=self.use_spectral_norm
            )
            for period in self.mpd_reshapes
        ]

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    resolution: List[int]
    d_mult: int = 1
    use_spectral_norm: bool = False

    def setup(self):
        if self.use_spectral_norm:
            raise NotImplementedError

        self.convs = [
            FlaxConvWithWeightNorm(1, 32 * self.d_mult, (3, 9)),
            FlaxConvWithWeightNorm(32 * self.d_mult, 32 * self.d_mult, (3, 9), (1, 2)),
            FlaxConvWithWeightNorm(32 * self.d_mult, 32 * self.d_mult, (3, 9), (1, 2)),
            FlaxConvWithWeightNorm(32 * self.d_mult, 32 * self.d_mult, (3, 9), (1, 2)),
            FlaxConvWithWeightNorm(32 * self.d_mult, 32 * self.d_mult, (3, 9)),
        ]
        self.conv_post = FlaxConvWithWeightNorm(32 * self.d_mult, 1, (3, 3))

        self.hann_window = self.variable(
            "params", "filter", lambda: jnp.hanning(self.resolution[2])
        )

    def __call__(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = jnp.expand_dims(x, -1)

        for l in self.convs:
            x = l(x)
            x = nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.reshape(1, -1)
        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = jnp.pad(
            x,
            (
                (0, 0),
                (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
                (0, 0),
            ),
            mode="reflect",
        )
        spec = spectrogram(
            x,
            pad=0,
            window=self.hann_window.value,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1.0,
            normalized=False,
            center=False,
            onesided=True,
        )
        spec = spec.transpose((0, 2, 1))
        return spec


class MultiResolutionDiscriminator(nn.Module):
    resolutions: Tuple[Tuple[int]]
    d_mult: int = 1
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert (
            len(self.resolutions) == 3
        ), f"MRD requires list of list with len=3, each element having a list with len=3. got {self.resolutions}"
        if self.use_spectral_norm:
            raise NotImplementedError
        self.discriminators = [
            DiscriminatorR(
                resolution, d_mult=self.d_mult, use_spectral_norm=self.use_spectral_norm
            )
            for resolution in self.resolutions
        ]

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiDiscriminator(nn.Module):
    mpd_reshapes: Tuple[int] = (2, 3, 5, 7, 11)
    resolutions: Tuple[Tuple[int]] = (
        ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)),
    )
    d_mult: int = 1
    use_spectral_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.use_spectral_norm:
            raise NotImplementedError
        self.mpd = MultiPeriodDiscriminator(
            self.mpd_reshapes,
            d_mult=self.d_mult,
            use_spectral_norm=self.use_spectral_norm,
            dtype=self.dtype,
        )
        self.mrd = MultiResolutionDiscriminator(
            self.resolutions[0],
            d_mult=self.d_mult,
            use_spectral_norm=self.use_spectral_norm,
            dtype=self.dtype,
        )

    def __call__(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.mrd(y, y_hat)

        return (
            y_df_hat_r,
            y_df_hat_g,
            fmap_f_r,
            fmap_f_g,
            y_ds_hat_r,
            y_ds_hat_g,
            fmap_s_r,
            fmap_s_g,
        )


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += jnp.mean(jnp.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = jnp.mean((1 - dr) ** 2)
        g_loss = jnp.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(jnp.array(r_loss, float))
        g_losses.append(jnp.array(g_loss, float))

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = jnp.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
