from typing import Optional

import tensorflow as tf
from keras import Model
from keras.callbacks import BackupAndRestore, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam

from birdseye.encoder.dataset import Dataloader
from birdseye.encoder.model import EncoderDecoder


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
