from typing import Optional

import tensorflow as tf
from keras import Model
from keras.callbacks import BackupAndRestore, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam

from birdseye.encoder.dataset import Dataloader


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
    # load_from_checkpoint: Optional[str] = None
):
    # if load_from_checkpoint is not None:
    #     model.load_weights(load_from_checkpoint)

    # Can I do all 4 at once?
    # front_data = Dataloader(batch_size, dataset_folder, "front")
    # rear_data = Dataloader(batch_size, dataset_folder, "rear")
    # passenger_side_data = Dataloader(batch_size, dataset_folder, "passenger_side")
    # driver_side_data = Dataloader(batch_size, dataset_folder, "driver_side")
    data = Dataloader(batch_size, dataset_folder, camera_type)

    # Validation stuff goes here later
    encoder_input = Input((256, 256, 3), name="encoder input")
    encoder_output = encoder(encoder_input)
    decoder_output = decoder(encoder_output)
    model = Model(encoder_input, decoder_output, name="encoder/decoder")
    print(model)
    print(model.summary())

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss=MeanAbsoluteError())

    # TODO - save the encoder and decoder weights separately,
    # and reload them separately, etc.
    callbacks = [
        ModelCheckpoint(f"{checkpoint_directory}/encoder_decoder.h5",
                        save_best_only=True),
        BackupAndRestore(backup_dir=f"{checkpoint_directory}/"),
        TensorBoard(log_dir=logdir, histogram_freq=1)
    ]

    model.fit(data, epochs=epochs, callbacks=callbacks)
