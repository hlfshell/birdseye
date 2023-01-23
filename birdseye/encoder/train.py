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
    encoder,
    decoder,
    discriminator,
    camera_type: str,
    epochs: int,
    batch_size: int,
    dataset_folder: str,
    validation_size: float = 0.25,
    checkpoint_directory: str = "./checkpoints/",
    logdir: str = "./logdir"
):
    data = Dataloader(batch_size, dataset_folder, camera_type)

    # Setup the encoder/decoder model
    model = EncoderDecoder(encoder, decoder)
    encoder_decoder_optimizer = Adam(learning_rate=0.005)
    model.compile(optimizer=encoder_decoder_optimizer)

    # Setup the discriminator model
    discriminator_optimizer = Adam()
    discriminator.compile(optimizer=discriminator_optimizer)

    mas_loss = MeanAbsoluteError()

    # TODO its a for loop for epoch AND dataset size
    for epoch in range(epochs):
        for index in range(len(data)):
            X, Y = data[index]

            with GradientTape(persistent=True) as tape:
                # Generate our fake outputs via the encoder/decoder combo
                # Then we convert it to 256 color range from (-1,1) range
                generated = model(X, training=True)
                # Convert the output to images for discriminator usage
                generated_imgs = np.asarray(
                    [np.asarray(tensor_to_image(img), dtype=np.uint8)
                     for img in generated])

                # Get the discriminator (non-trained) results from the
                # generated # images
                discriminator_output = discriminator(generated_imgs)

                # Train on 50/50 fake and real
                fake_output = discriminator(generated_imgs, training=True)
                real_output = discriminator(X, training=True)

                # Take the average loss across all of them.
                fake_loss = fake_output
                real_loss = 1 - real_output
                discriminator_loss_combined = 0.5 * (fake_loss + real_loss)

                # Now train the generator for this epoch
                mas_batch_loss = mas_loss(Y, generated)
                encoder_decoder_combined_loss = mas_batch_loss + \
                    (1 - discriminator_output)

            encoder_decoder_gradients = tape.gradient(
                encoder_decoder_combined_loss, model.trainable_weights)
            discriminator_gradients = tape.gradient(
                discriminator_loss_combined, discriminator.trainable_weights)

            # Apply gradients from losses
            encoder_decoder_optimizer.apply_gradients(
                zip(encoder_decoder_gradients, model.trainable_weights))
            discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, discriminator.trainable_weights))

            raise "wait really?"  # stopping here because I have more to do
            # but i'm stuck above so....


def train_birdseye_encoder_with_discriminator2(
    camera_type: str,
    epochs: int,
    batch_size: int,
    dataset_folder: str,
    validation_size: float = 0.25,
    checkpoint_directory: str = "./checkpoints/",
    logdir: str = "./logdir",
    critic_trains_per_generator: int = 5,
):
    data = Dataloader(batch_size, dataset_folder, camera_type)

    # Setup the encoder/decoder model
    model = EncoderDecoder(Encoder(), Decoder())
    # critic model
    critic = Critic()

    trainable_toggle(model, False)

    # Setup the discriminator
    real_images = Input(shape=(256, 256, 3))
    generated_images_input = Input(shape=(256, 256, 3))
    fake_images = model(generated_images_input)
    interpolated_images = InterpolateLayer(
        batch_size)([real_images, fake_images])
    interpolated_output = critic(interpolated_images)

    # Discriminator is used against real and fake images
    real = critic(real_images)
    fake = critic(fake_images)

    # Build the loss function next
    partial_gp_loss = partial(gradient_penalty_loss_interpolated,
                              interpolated_samples=interpolated_images)
    partial_gp_loss.__name__ = "gradient_penalty"  # Required by Keras

    adversarial = Model(inputs=[real_images, generated_images_input], outputs=[
        real, fake, interpolated_output], name="adversarial")
    adversarial.compile(
        loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
        optimizer=Adam(learning_rate=0.0002),
        loss_weights=[1, 1, 10]  # partial gradient loss is 10x others
    )

    trainable_toggle(critic, True)

    valid = np.ones((batch_size, 1), dtype=np.float32)
    fake = -1 * np.ones((batch_size, 1), dtype=np.float32)
    ignored = np.zeros((batch_size, 1), dtype=np.float32)

    total_epochs = 0
    critic_epoch = 0
    generator_epoch = 0

    for epoch in range(epochs):
        critic_avg_losses = []
        generator_avg_losses = []

        # Make the critic trainable, the generator not trainable
        trainable_toggle(critic, True)
        trainable_toggle(model, False)

        for _ in range(critic_trains_per_generator):
            for index in range(len(data)):
                inputs, real = data[index]
                # Generate the output with the generator
                # generated = model(inputs, training=False)

                critic_loss = adversarial.train_on_batch(
                    [real, real], [valid, fake, ignored])
                raise "fucking finally"
                critic_avg_losses.append(critic_loss)

                print(
                    f"Epoch {epoch+1} | " +
                    f"Critic epoch {critic_epoch+1} | " +
                    f"Batch {index+1}/{len(data)} | " +
                    f"Batch Loss: {critic_loss} | " +
                    f"Average Loss: {sum(critic_avg_losses)/len(critic_avg_losses)}"
                )
            critic_epoch += 1
            print("")
            raise "huh?"


def train_birdseye_encoder_with_discriminator3(
    camera_type: str,
    epochs: int,
    batch_size: int,
    dataset_folder: str,
    validation_size: float = 0.25,
    checkpoint_directory: str = "./checkpoints/",
    logdir: str = "./logdir",
    critic_trains_per_generator: int = 5,
):
    data = Dataloader(batch_size, dataset_folder, camera_type)
    Path(f"{checkpoint_directory}/models").mkdir(parents=True, exist_ok=True)

    # Setup the encoder/decoder model
    model = EncoderDecoder(Encoder(), Decoder())
    # Critic model
    critic = Critic()

    total_epochs = 0
    critic_epoch = 0
    generator_epoch = 0

    critic.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.0002)
    )
    model.compile(
        loss=partial(generator_loss, base_loss=MeanSquaredError(),
                     discriminator=critic),
        optimizer=Adam(learning_rate=0.0002)
    )
    model.run_eagerly = True

    real = np.ones((batch_size, 1), dtype=np.float32)
    fake = -1*np.ones((batch_size, 1), dtype=np.float32)

    for epoch in range(epochs):
        critic_avg_losses = []
        generator_avg_losses = []

        # Make the critic trainable, the generator not trainable
        trainable_toggle(critic, True)
        trainable_toggle(model, False)

        for _ in range(critic_trains_per_generator):
            for index in range(len(data)):
                raw_imgs, expected_out = data[index]
                # Generate the output with the generator
                generated = model(raw_imgs, training=False)

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
                    f"Critic epoch {critic_epoch+1} | " +
                    f"Batch {index+1}/{len(data)} | " +
                    f"Batch Loss: {loss_combined} | " +
                    f"Average Loss: {sum(critic_avg_losses)/len(critic_avg_losses)}",
                    end="\r"
                )
            critic_epoch += 1
        print("")

        trainable_toggle(critic, False)
        trainable_toggle(model, True)

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

        # Save the models by epoch
        critic.save_weights(
            f"{checkpoint_directory}/models/critic_{epoch+1}.h5")
        model.save_weights(
            f"{checkpoint_directory}/models/encoderdecoder_{epoch+1}.h5")

    return model


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
