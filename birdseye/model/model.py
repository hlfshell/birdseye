from tensorflow import keras
from keras.applications import ResNet50
from keras.layers import Conv2D, merge


def Model(input_shape=(255,255,3)):

    # Each input image is fed through ResNet50 with
    # locked layers to start
    resnet_front = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_front.layers:
        layer.trainable = False

    resnet_rear = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_rear.layers:
        layer.trainable = False
    
    resnet_passenger_side = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_passenger_side.layers:
        layer.trainable = False

    resnet_driver_side = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_driver_side.layers:
        layer.trainable = False
    
    # Combine the output of all of these layers
    merged = merge([resnet_front, resnet_rear, resnet_passenger_side, resnet_driver_side], mode='concat')

    # A few CNNs over the combined input
    net = Conv2D(128, kernel_size=3, activation='relu')(merge)
    net = Conv2D(64, kernel_size=3, activation='relu')(net)
    net = Conv2D(32, kernel_size=3, activation='relu')(net)

    # Up Sample back
    