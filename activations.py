import flax.linen as nn
import jax
import jax.numpy as jnp


class Snake(nn.Module):
    in_features: int
    alpha_value: float = 1.0
    alpha_trainable: bool = True
    alpha_logscale: bool = False

    def setup(self):
        if self.alpha_logscale:
            self.alpha = self.param(
                "alpha", jax.nn.initializers.zeros, (self.in_features,)
            )
        else:
            self.alpha = self.param(
                "alpha", jax.nn.initializers.ones, (self.in_features,)
            )

        self.alpha = self.alpha * self.alpha_value

    def __call__(self, x):
        alpha = self.alpha.reshape(1, 1, -1)
        if self.alpha_logscale:
            alpha = jnp.exp(alpha)
        x = x + (1.0 / (alpha + 1e-9)) * jnp.power(jnp.sin(x * alpha), 2)
        return x


class SnakeBeta(nn.Module):
    in_features: int
    alpha_value: float = 1.0
    alpha_trainable: bool = True
    alpha_logscale: bool = False

    def setup(self):
        if self.alpha_logscale:
            self.alpha = self.param(
                "alpha", jax.nn.initializers.zeros, (self.in_features,)
            )
            self.beta = self.param(
                "beta", jax.nn.initializers.zeros, (self.in_features,)
            )
        else:
            self.alpha = self.param(
                "alpha", jax.nn.initializers.ones, (self.in_features,)
            )
            self.beta = self.param(
                "beta", jax.nn.initializers.zeros, (self.in_features,)
            )

        self.alpha = self.alpha * self.alpha_value
        self.beta = self.beta * self.alpha_value

    def __call__(self, x):
        alpha = self.alpha.reshape(1, 1, -1)
        beta = self.beta.reshape(1, 1, -1)
        if self.alpha_logscale:
            alpha = jnp.exp(alpha)
            beta = jnp.exp(beta)
        x = x + (1.0 / (beta + 1e-9)) * jnp.power(jnp.sin(x * alpha), 2)
        return x
