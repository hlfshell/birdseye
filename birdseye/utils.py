import numpy as np

from PIL import Image
from typing import Tuple





def image_to_tensor(img: Image) -> np.ndarray:
    """
    image_to_tensor: takes an incoming image and returns it as an equvialently
        shaped tensor where all values have been mapped from its original
        (0,255) to (-1,1).

        :param image: a PIL image
        :return np.ndarray - the transformed tensor
    """
    # First ensure that we have an RGB and not an RGBA image - we don't want
    # the extra channel
    img = img.convert("RGB")

    tensor = np.asarray(img, dtype=np.float32)
    mapper = np.vectorize(values_mapper((0,255), (-1,1)))
    tensor = mapper(tensor)

    return tensor


def tensor_to_image(tensor: np.ndarray) -> Image:
    """
    tensor_to_image: takes a tensor/np.ndarray and returns a PIL image with
        values transformed from an assumed (-1,1) to (0,255)

        :param tensor: np.ndarray or tensor of shape (255,255,3)
        :return Image: resulting PIL image
    """
    mapper = np.vectorize(values_mapper((-1,1), (0,255)))
    tensor = mapper(tensor).astype(np.uint8)

    print(tensor)

    return Image.fromarray(tensor, 'RGB')
    

def resize_semantic_labels(tensor: np.ndarray, size: Tuple) -> np.ndarray:
    """
    resize_semantic_labels: takes a tensor/numpy array and resizes it to the
        desired size. This has to be handled different from standard resizing
        due to the nature of the labels - we can't do any interpolation or
        averaging of the values as they are markers of a selected label, not
        a value that has particular meaning. Put another way - if a category 2
        pixel is near a category 3 pixel, we gain no meaning from a value of 2.5.

        :param size: Tuple of size (x,y, categories)
        :return np.ndarray -> The resized tensor
    """
    pass


def values_mapper(from_range: Tuple[float, float], to_range: Tuple[float, float])-> callable:
    """
    values: takes a from range and to range, and returns a function that maps
        a value between the ranges.

    :param from_range: Tuple[float, float] - the range the value currently
        resides, in (min,max) form
    :param to_range: Tuple[float, float] - the range the value should be mapped
        to, in (min, max) form
    :return callable: a function that maps from one range to the other
    """
    def mapper(value):
        return (((value - from_range[0]) / (from_range[1] - from_range[0])) * \
        (to_range[1]-to_range[0])) + to_range[0]
    
    return mapper
