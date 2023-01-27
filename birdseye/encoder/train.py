import os
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras.callbacks import BackupAndRestore, ModelCheckpoint, TensorBoard
from keras.layers import Input, Layer
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.optimizers import Adam
from tensorflow import GradientTape

from birdseye.encoder.dataset import Dataloader
from birdseye.encoder.model import (
    Critic,
    Decoder,
    Discriminator,
    Encoder,
    EncoderDecoder,
)
from birdseye.utils import tensor_to_image


def train_birdseye_encoder(
    encoder,
    decoder,
    camera_type: str,
    epochs: int,
    batch_size: int,
    dataset_folder: str,
    validation_size: float = 0.25,
    checkpoint_directory: str = "./checkpoints/",
    logdir: str = "./logdir",
):
    data = Dataloader(batch_size, dataset_folder, camera_type)
    model = EncoderDecoder(encoder, decoder)

    print(model.summary())

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss=MeanAbsoluteError())

    # TODO - save the encoder and decoder weights separately,
    # and reload them separately, etc.
    callbacks = [
        ModelCheckpoint(f"{checkpoint_directory}/encoder_decoder.h5",
                        save_best_only=False),
        BackupAndRestore(backup_dir=f"{checkpoint_directory}/"),
        TensorBoard(log_dir=logdir, histogram_freq=1)
    ]

    model.fit(data, epochs=epochs, callbacks=callbacks)


def train_birdseye_encoder_with_discriminator(
    camera_type: str,
    epochs: int,
    batch_size: int,
    dataset_folder: str,
    validation_size: float = 0.25,
    checkpoint_directory: str = "./checkpoints/",
    critic_trains_per_generator: int = 5,
):
    data = Dataloader(batch_size, dataset_folder, camera_type)
    # Split off 15% of our dataset for validation testing
    validation_data = data.split_off_percentage(0.15)

    # Ensure that our checkpoints directory is set
    Path(f"{checkpoint_directory}/").mkdir(parents=True, exist_ok=True)

    # Setup the encoder/decoder model
    model = EncoderDecoder(Encoder(), Decoder())
    # Critic model
    critic = Critic()

    critic.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.0001)
    )
    model.compile(
        loss=partial(generator_loss, base_loss=MeanSquaredError(),
                     discriminator=critic),
        optimizer=Adam(learning_rate=0.0001)
    )

    # If we have a checkpoints directory, load the last step and resume
    # training from where we left off
    starting_epoch = 0
    critic_epochs = 0
    if len(os.listdir(checkpoint_directory)) >= 1:
        model, critic, epoch = load_encoder_decoder_critic_checkpoint(
            checkpoint_directory,
            model,
            critic,
        )
        starting_epoch = epoch
        critic_epochs = epoch * critic_trains_per_generator
        print(f"Resuming training from epoch {epoch}")

    # Results precoded for real/generations
    # 1 for real, -1 for generated
    real = np.ones((batch_size, 1), dtype=np.float32)
    fake = -1*np.ones((batch_size, 1), dtype=np.float32)

    for epoch in range(starting_epoch, epochs):
        critic_avg_losses = []
        generator_avg_losses = []

        # Make the critic trainable, the generator not trainable
        trainable_toggle(critic, True)
        trainable_toggle(model, False)
        # Turn off data augmentation for the critic steps
        data.apply_data_augmentation = False

        # Train the critic multiple times per generator training epoch
        for _ in range(critic_trains_per_generator):
            for index in range(len(data)):
                raw_imgs, expected_out = data[index]
                # Generate the output with the generator
                generated = model(raw_imgs, training=False)

                # Calculate our losses
                critic_loss_real = critic.train_on_batch(
                    expected_out, real
                )
                critic_loss_fake = critic.train_on_batch(
                    generated, fake
                )
                loss_combined = critic_loss_real + critic_loss_fake

                critic_avg_losses.append(loss_combined)

                print(
                    f"Epoch {epoch+1} | " +
                    f"Critic epoch {critic_epochs+1} | " +
                    f"Batch {index+1}/{len(data)} | " +
                    f"Batch Loss: {loss_combined} | " +
                    f"Average Loss: {sum(critic_avg_losses)/len(critic_avg_losses)}",
                    end="\r"
                )
            critic_epochs += 1
        print("")

        # Swap the trainability - critic is no longer trainable, generator
        # will be trained now
        trainable_toggle(critic, False)
        trainable_toggle(model, True)
        # Turn on data augmentation for the generator steps
        data.apply_data_augmentation = True

        for index in range(len(data)):
            raw_imgs, expected_out = data[index]

            loss = model.train_on_batch(raw_imgs, expected_out)
            generator_avg_losses.append(loss)

            print(
                f"Epoch {epoch+1} | " +
                f"Generator epoch {epoch+1} | " +
                f"Batch {index+1}/{len(data)} | " +
                f"Batch Loss: {loss} | " +
                f"Average Loss: {sum(generator_avg_losses)/len(generator_avg_losses)}",
                end="\r"
            )

        print("")

        # Validation run
        trainable_toggle(model, False)
        validation_losses = []
        for index in range(len(validation_data)):
            raw_imgs, expected_out = data[index]
            generated = model(raw_imgs)
            loss = generator_loss(expected_out, generated,
                                  MeanAbsoluteError(), critic)
            validation_losses.append(loss[0])

        print(
            f"Validation loss for generator for epoch {epoch+1} " +
            f"is {sum(validation_losses)/len(validation_losses)}"
        )

        # Save our progress thusfar
        critic.save_weights(
            f"{checkpoint_directory}/critic_{epoch+1}.h5")
        model.save_weights(
            f"{checkpoint_directory}/generator_{epoch+1}.h5")

    return model


def load_encoder_decoder_critic_checkpoint(checkpoint_dir: str, generator, critic):
    files = os.listdir(checkpoint_dir)
    files = [file for file in files if file.endswith(
        ".h5") and file.startswith("critic_")]
    highest_epoch = 0
    for filename in files:
        splits = filename.split("_")
        epoch = splits[1].split(".")[0]
        if int(epoch) > highest_epoch:
            highest_epoch = int(epoch)

    # The trainable setting for models must match the way they were set
    # at time of save. Thus, since both models are set to not trainagble
    # prior to saving, we do that here now when loading weights.
    trainable_toggle(generator, False)
    trainable_toggle(critic, False)

    generator.load_weights(os.path.join(
        checkpoint_dir, f"generator_{highest_epoch}.h5"))
    critic.load_weights(os.path.join(
        checkpoint_dir, f"critic_{highest_epoch}.h5"))

    return generator, critic, highest_epoch


class InterpolateLayer(Layer):
    """
    This layer returns a uniform random mixture of fake/real combined,
    limiting it to our desired batch size.
    """

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def wasserstein_loss(y_true, y_pred):
    return -K.mean(y_true*y_pred)


def wasserstein_discriminator_loss(y_true, y_pred, discriminator):
    discriminator_out = discriminator(y_pred)
    fake = -1 * np.ones((len(y_true,), 1))
    return -K.mean(fake * discriminator_out)


def generator_loss(y_true, y_pred, base_loss, discriminator):
    discriminator_preds = discriminator(y_pred)
    discriminator_loss = 1 - discriminator_preds
    base_loss = base_loss(y_true, y_pred)
    # print(" ")
    # print("==")
    # print(discriminator_preds)
    # print(discriminator_loss)
    # print(base_loss)
    # print(discriminator_loss + base_loss)
    # print("==")
    # print(" ")
    return discriminator_loss + base_loss


def gradient_penalty_loss_interpolated(y_true, y_pred, interpolated_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, interpolated_samples)[0]

    # Compute the Euclidean norm by squaring
    gradients_squared = K.square(gradients)
    # Sum over all rows
    gradients_square_sum = K.sum(
        gradients_squared, axis=np.arange(1, len(gradients_squared.shape)))
    # Square root of the gradient
    gradient_l2_norm = K.sqrt(gradients_square_sum)
    # Lambda computation - (1 - ||grad||)^2 for each single sample
    gradient_penalty = K.square(1-gradient_l2_norm)
    # Finally, return the mean loss over all batch samples
    return K.mean(gradient_penalty)


# def gradient_penalty_loss(y_true, y_pred):
#     gradients = K.gradients(y_pred, interpolated_samples)[0]

#     # Compute the Euclidean norm by squaring
#     gradients_squared = K.square(gradients)
#     # Sum over all rows
#     gradients_square_sum = K.sum(
#         gradients_squared, axis=np.arange(1, len(gradients_squared.shape)))
#     # Square root of the gradient
#     gradient_l2_norm = K.sqrt(gradients_square_sum)
#     # Lambda computation - (1 - ||grad||)^2 for each single sample
#     gradient_penalty = K.square(1-gradient_l2_norm)
#     # Finally, return the mean loss over all batch samples
#     return K.mean(gradient_penalty)


def trainable_toggle(model, trainable: bool):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
