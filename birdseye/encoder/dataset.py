from __future__ import annotations

import os
from random import shuffle
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image
from tensorflow import keras

from birdseye.utils import image_to_tensor, split_dataset_filename


class Dataloader(keras.utils.Sequence):

    def __init__(
        self,
        batch_size: int,
        dataset_folder: str,
        camera_type: str,
        data: Optional[Tuple[str]] = None
    ):
        self.batch_size = batch_size
        self.camera_type = camera_type
        self.data_folder = dataset_folder
        
        if data is None:
            self._init_data_(dataset_folder)
        else:
            self.data = data
        
        shuffle(self.data)

    def _init_data_(self, dataset_folder: str):
        self.data = []
        input_folder = os.path.join(dataset_folder, "input")
        files = os.listdir(input_folder)
        files = [file in files if self.camera_type in file]

        for file in files:
            runid, frame, _ = split_dataset_filename(file)
            id = f"{runid}_{frame}"
            
            self.data.append[id]
    
    def split_off_percentage(
        self,
        percent: float,
        batch_size: Optional[float] = None
    ):
        if batch_size is None:
            batch_size = self.batch_size
        
        shuffle(self.data)
        split_index = int(len(self.data)*percent)
        split = self.data[0:split_index]
        self.data = self.data[split_index:]

        return Dataloader(batch_size, self.data_folder, self.camera_type, data=split)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size
    
    def __getitem__(self, start):
        index = start * self.batch_size
        data = self.data[index: index+self.batch_size]

        inputs = np.zeros((self.batch_size, 256, 256, 3))

        for file in data:
            filepath = os.path.join(self.data_folder, "input", file)
            img = Image.open(filepath).resize((256,256)).convert('RGB')
            inputs[0] = np.asarray(img, dtype=np.float32)

        targets = np.copy(inputs)

        return inputs, targets