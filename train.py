import argparse
import json

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import jax_utils
from flax.training import checkpoints, train_state
from flax.training.common_utils import shard

from meldataset import mel_spectrogram, get_dataset
from models import (
    BigVGAN,
    MultiDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from utils import AttrDict


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", default="data/train")
    parser.add_argument("--test_dir", default="data/test")

    parser.add_argument("--checkpoint_path", default="exp/bigvgan")
    parser.add_argument("--config", default="")

    parser.add_argument("--training_steps", default=100000, type=int)
    parser.add_argument("--checkpoint_interval", default=50000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    # parser.add_argument("--validation_interval", default=50000, type=int)

    parser.add_argument("--fine_tuning", default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        json_config = json.load(f)
    h = AttrDict(json_config)

    generator = BigVGAN(
        num_mels=h.num_mels,
        resblock_kernel_sizes=h.resblock_kernel_sizes,
        resblock_dilation_sizes=h.resblock_dilation_sizes,
        upsample_rates=h.upsample_rates,
        upsample_initial_channel=h.upsample_initial_channel,
        upsample_kernel_sizes=h.upsample_kernel_sizes,
        activation=h.activation,
        alpha_logscale=h.snake_logscale,
    )
    discriminator = MultiDiscriminator(
        mpd_reshapes=h.mpd_reshapes,
        resolutions=h.resolutions,
        d_mult=h.discriminator_channel_mult,
    )

    expotential_lr_schedule_fn = optax.exponential_decay(
        init_value=h["learning_rate"],
        transition_steps=h["lr_decay_steps"],
        decay_rate=h["lr_decay"],
    )

    optim_g = optax.adamw(
        learning_rate=expotential_lr_schedule_fn, b1=h["adam_b1"], b2=h["adam_b2"]
    )
    optim_d = optax.adamw(
        learning_rate=expotential_lr_schedule_fn, b1=h["adam_b1"], b2=h["adam_b2"]
    )

    generator_state = train_state.TrainState.create(
        apply_fn=generator.__call__, params=generator.params, tx=optim_g
    )
    discriminator_state = train_state.TrainState.create(
        apply_fn=discriminator.__call__, params=discriminator.params, tx=optim_d
    )

    trainset = get_dataset(a.train_dir, h.batch_size, h.num_mels)

    def train_step(state_g, state_d, batch):
        def g_loss_fn(params):
            x, y = batch["x"], batch["y"]
            y_g_hat = state_g.apply_fn(params, x)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat,
                n_fft=h.n_fft,
                num_mels=h.num_mels,
                sampling_rate=h.sampling_rate,
                hop_size=h.hop_size,
                win_size=h.win_size,
                fmin=0,
                fmax=None,
            )

            (
                y_df_hat_r,
                y_df_hat_g,
                fmap_f_r,
                fmap_f_g,
                y_ds_hat_r,
                y_ds_hat_g,
                fmap_s_r,
                fmap_s_g,
            ) = state_d.apply_fn(state_d.params, jnp.expand_dims(y, -1), y_g_hat)

            loss_mel = jnp.mean(jnp.abs(y_g_hat_mel - x)) * 45

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            return loss_gen_all, (
                loss_mel,
                loss_fm_f,
                loss_gen_f,
                loss_fm_s,
                loss_gen_s,
            )

        def d_loss_fn(params):
            x, y = batch["x"], batch["y"]
            y_g_hat = state_g.apply_fn(state_g.params, x)
            (
                y_df_hat_r,
                y_df_hat_g,
                fmap_f_r,
                fmap_f_g,
                y_ds_hat_r,
                y_ds_hat_g,
                fmap_s_r,
                fmap_s_g,
            ) = state_d.apply_fn(params, jnp.expand_dims(y, -1), y_g_hat)

            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f

            return loss_disc_all, (loss_disc_f, loss_disc_s)

        g_grad_fn = jax.value_and_grad(g_loss_fn, has_aux=True)
        (
            loss_gen_all,
            (loss_mel, loss_fm_f, loss_gen_f, loss_fm_s, loss_gen_s),
        ), g_grads = g_grad_fn(state_g.params)
        g_grads = jax.lax.pmean(g_grads, "batch")

        new_state_g = state_g.apply_gradients(grads=g_grads)

        d_grad_fn = jax.value_and_grad(d_loss_fn, has_aux=True)
        (loss_disc_all, (loss_disc_f, loss_disc_s)), d_grads = d_grad_fn(state_d.params)
        d_grads = jax.lax.pmean(d_grads, "batch")

        new_state_d = state_d.apply_gradients(grads=d_grads)

        metrics = {
            "loss/gen/total": loss_gen_all,
            "loss/gen/mel": loss_mel,
            "loss/gen/fm_f": loss_fm_f,
            "loss/gen/gen_f": loss_gen_f,
            "loss/gen/fm_s": loss_fm_s,
            "loss/gen/gen_s": loss_gen_s,
            "loss/disc/total": loss_disc_all,
            "loss/disc/f": loss_disc_f,
            "loss/disc/s": loss_disc_s,
        }

        return new_state_g, new_state_d, metrics

    p_train_step = jax.pmap(
        train_step,
        axis_name="batch",
        donate_argnums=(
            0,
            1,
        ),
    )

    generator_state = jax_utils.replicate(generator_state)
    discriminator_state = jax_utils.replicate(discriminator_state)

    step = 0
    while step < h.training_steps:
        for (x, y) in trainset:
            batch = {"x": x.numpy(), "y": y.numpy()}
            batch = shard(batch)

            generator_state, discriminator_state, metrics = p_train_step(
                generator_state, discriminator_state, batch
            )

            step += 1
            if step % a.summary_interval == 0:
                metrics = jax_utils.unreplicate(metrics)
                metrics = jax.tree_map(lambda x: float(x), metrics)

                wandb.log(metrics, step=step)

            if step % a.checkpoint_interval == 0:
                if jax.process_index() == 0:
                    g_params = jax.device_get(
                        jax.tree_util.tree_map(lambda x: x[0], generator_state.params)
                    )
                    d_params = jax.device_get(
                        jax.tree_util.tree_map(
                            lambda x: x[0], discriminator_state.params
                        )
                    )

                    custom_ckpt = {
                        "generator": g_params,
                        "discriminator": d_params,
                        "step": step,
                    }
                    checkpoints.save_checkpoint(a.output_dir, custom_ckpt, step, keep=3)


if __name__ == "__main__":
    main()
