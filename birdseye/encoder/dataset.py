from __future__ import annotations

import os
from random import randint, shuffle, uniform
from typing import List, Optional

import numpy as np
import skimage
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from tensorflow import keras

from birdseye.utils import image_to_tensor


class Dataloader(keras.utils.Sequence):

    def __init__(
        self,
        batch_size: int,
        dataset_folder: str,
        camera_type: str,
        data_augmentation: bool = False,
        data: Optional[List[str]] = None
    ):
        self.batch_size = batch_size
        self.camera_type = camera_type
        self.data_folder = dataset_folder
        self.apply_data_augmentation = data_augmentation

        if data is None:
            self._init_data_(dataset_folder)
        else:
            self.data = data

        shuffle(self.data)

    def _init_data_(self, dataset_folder: str):
        self.data = []
        input_folder = os.path.join(dataset_folder, "input")
        files = os.listdir(input_folder)
        self.data = [file for file in files if self.camera_type in file]

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def __getitem__(self, start):
        index = start * self.batch_size
        data = self.data[index: index+self.batch_size]

        inputs = np.zeros((self.batch_size, 256, 256, 3))
        targets = np.zeros_like(inputs)

        for index, file in enumerate(data):
            filepath = os.path.join(self.data_folder, "input", file)
            img = Image.open(filepath).resize((256, 256)).convert('RGB')
            targets[index] = image_to_tensor(img)
            if self.apply_data_augmentation:
                img = self.data_augmentation(img)
            inputs[index] = np.asarray(img, dtype=np.float32)

        return inputs, targets

    def data_augmentation(self, img: Image) -> Image:
        # Play with brightness +/- 40%
        brightness_adjust = 1.0 + uniform(-.40, 0.40)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_adjust)

        # Play with contrast - 1.0 is original image. We'll be willing to go
        # a bit lower at 40% less contrast
        contrast_adjust = 1 - uniform(0.0, 0.40)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_adjust)

        # Add blurring - the blur radius affects the blur so we'll go
        # from 0.0 to 2.5
        blur_radius = uniform(0.0, 2.5)
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

        # Add noise to the image
        noise_amount = uniform(0.0, 0.07)
        img_arr = np.asarray(img, dtype="uint8")
        img_arr = 255 * \
            skimage.util.random_noise(
                img_arr, mode='salt', amount=noise_amount)
        img = Image.fromarray(np.uint8(img_arr))

        # Cutout - random black squares over some section of the image
        # Random if we even apply it as well
        if randint(0, 1):
            draw = ImageDraw.Draw(img)
            # Determine the size of the triangle. No more than 10% of the image
            size = int(img.size[0] * 0.10)
            x = randint(0, img.size[0]-1)  # random x position
            y = randint(0, img.size[0]-1)  # random y position
            start_corner = (x, y)
            end_corner = (
                min(x+size, img.size[0]-1), min(y+size, img.size[0]-1))
            draw.rectangle([start_corner, end_corner], fill="black")

        return img

    def split_off_percentage(
        self,
        percent: float,
        batch_size: Optional[float] = None,
        data_augmentation: Optional[bool] = None
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
            data_augmentation = self.apply_data_augmentation

        shuffle(self.data)
        split = self.data[0:int(len(self.data)*percent)]
        self.data = self.data[int(len(self.data)*percent):]

        return Dataloader(batch_size, "", self.camera_type, data_augmentation=data_augmentation, data=split)
