from typing import Optional

from keras.callbacks import BackupAndRestore, ModelCheckpoint, TensorBoard

from birdseye.dataset import Dataloader


def train_birdseye(
    model,
    epochs: int,
    batch_size: int,
    dataset_folder: str,
    validation_size: float = 0.25,
    checkpoint_directory: str = "./checkpoints/",
    logdir: str = "./logdir",
    load_from_checkpoint: Optional[str] = None
):
    if load_from_checkpoint is not None:
        model.load_weights(load_from_checkpoint)

    train_data = Dataloader(
        batch_size=batch_size,
        dataset_folder=dataset_folder,
        data_augmentation=False
    )
    validation_data = train_data.split_off_percentage(validation_size)

    model.compile(optimizer='adam', loss='mse')

    callbacks = [
        ModelCheckpoint(f"{checkpoint_directory}/birdseye.h5",
                        save_best_only=True),
        BackupAndRestore(backup_dir=f"{checkpoint_directory}/"),
        TensorBoard(log_dir=logdir, histogram_freq=1)
    ]

    model.fit(train_data, epochs=epochs,
              validation_data=validation_data, callbacks=callbacks)

    return model
