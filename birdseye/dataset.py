from __future__ import annotations

import os
from random import shuffle
from typing import Dict, Optional

import numpy as np
from PIL import Image
from tensorflow import keras
from utils import image_to_tensor, split_dataset_filename


class Dataloader(keras.utils.Sequence):

    def __init__(
        self,
        batch_size: int,
        dataset_folder: str,
        data_augmentation: bool = False,
        data: Optional[Dict[str, Dict[str, str]]] = None
    ):
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation

        # If we've not been passed data, we have to init here
        if data is None:
            self._init_data(dataset_folder)
        else:
            self.data = data

        self.ids = shuffle(self.data.keys())

    def _init_data(self, dataset_folder):
        # Self.data will contain all images for each set, input *and* output
        self.data = {}

        # First we move through the dataset and group all of the input
        # images together in the correct order
        input_folder = os.path.join(dataset_folder, "input")

        for file in os.listdir(input_folder):
            # png's only
            if not file.endswith(".png"):
                continue

            runid, frame, camera_type = split_dataset_filename(file)
            id = f"{runid}_{frame}"
            if id not in self.data:
                self.data[id] = {}

            self.data[id][camera_type] = os.path.join(input_folder, file)

        # Now grab all the target outputs
        overhead_folder = os.path.join(dataset_folder, "overhead")

        for file in os.listdir(overhead_folder):
            # png's only
            if not file.endswith(".png"):
                continue

            runid, frame, camera_type = split_dataset_filename(file)
            id = f"{runid}_{frame}"
            if id not in self.data:
                # If we haven't seen it yet, we don't have the inputs for
                # it; therefore ignore it
                continue

            self.data[id][camera_type] = os.path.join(overhead_folder, file)

        # Go over all the data and remove any sets that do not have 5 images
        for id in self.data:
            if len(self.data[id]) != 5:
                del self.data[id]

    def split_off_percentage(
        self,
        percent: float,
        batch_size: Optional[float],
        data_augmentation: Optional[bool]
    ) -> Dataloader:
        """
        split_off_percentage: accepts a given percentage, and then returns a
            random subset of this parent's data. This also removes the data
            from the parent. This is to be used for splitting validation
            datasets, for instance.

        :param percent: float - [0,1]
        :param batch_size: None/float - if None, adopts the batch_size of the
            parent
        :param data_augmentation: None/bool - if None, adopts the
            data_augmentation of the parent
        :return Dataloader
        """
        if batch_size is None:
            batch_size = self.batch_size
        if data_augmentation is None:
            data_augmentation = self.data_augmentation

        sets = self.data.keys()
        split = shuffle(sets)[int(len(sets)*percent)]
        data = {}

        for id in split:
            data[id] = self.data[id]
            del self.data[id]

        return Dataloader(batch_size, "", data_augmentation, data=data)

    def apply_augmentation(self, img: Image) -> Image:
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, start):
        # Get our target ids
        index = start * self.batch_size
        ids = self.ids[index: index+self.batch_size]
        current_sets = [self.data[id] for id in ids]

        tensor_size = (self.batch_size, 256, 256, 3)
        overhead_batch = np.zeros(tensor_size)
        front_batch = np.zeros(tensor_size)
        rear_batch = np.zeros(tensor_size)
        passenger_side_batch = np.zeros(tensor_size)
        driver_side_batch = np.zeros(tensor_size)

        for index, set in enumerate(current_sets):
            # Overhead target
            img = Image.open(set["birdseye"])
            overhead_batch[index] = image_to_tensor(img)

            # Front input
            img = Image.open(set["front"])
            if self.data_augmentation:
                img = self.apply_augmentation(img)
            front_batch[index] = np.asarray(img, dtype=np.float32)

            # Rear input
            img = Image.open(set["rear"])
            if self.data_augmentation:
                img = self.apply_augmentation(img)
            rear_batch[index] = np.asarray(img, dtype=np.float32)

            # Passenger side input
            img = Image.open(set["passenger_side"])
            if self.data_augmentation:
                img = self.apply_augmentation(img)
            passenger_side_batch[index] = np.asarray(img, dtype=np.float32)

            # Driver side input
            img = Image.open(set["driver_side"])
            if self.data_augmentation:
                img = self.apply_augmentation(img)
            driver_side_batch[index] = np.asarray(img, dtype=np.float32)

        return front_batch, rear_batch, passenger_side_batch, \
            driver_side_batch, overhead_batch
