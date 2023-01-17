from typing import Optional

import numpy as np
import tensorflow as tf
from keras import Model
from keras.callbacks import BackupAndRestore, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam
from tensorflow import GradientTape

from birdseye.encoder.dataset import Dataloader
from birdseye.encoder.model import Discriminator, EncoderDecoder
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
