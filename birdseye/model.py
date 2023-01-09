from tensorflow import keras
from keras.applications import ResNet50
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2DTranspose,
    UpSampling2D
)


def BirdsEye(input_shape=(255, 255, 3)):
    inputs = keras.Input(shape=input_shape)

    # Each input image is fed through ResNet50 with
    # locked layers to start
    resnet_front = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_front.layers:
        layer.trainable = False
    resnet_front._name = "resnet_front"
    resnet_front = resnet_front(inputs)

    resnet_rear = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_rear.layers:
        layer.trainable = False
    resnet_rear._name = "resnet_rear"
    resnet_rear = resnet_rear(inputs)

    resnet_passenger_side = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_passenger_side.layers:
        layer.trainable = False
    resnet_passenger_side._name = "resnet_passenger_side"
    resnet_passenger_side = resnet_passenger_side(inputs)

    resnet_driver_side = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape

    )
    for layer in resnet_driver_side.layers:
        layer.trainable = False
    resnet_driver_side._name = "resnet_driver_side"
    resnet_driver_side = resnet_driver_side(inputs)

    # Output of a single ResNet50 at our input is (batch, 8, 8, 2048)
    # So our merged output is (batch, 8, 8, 8192)
    net = Concatenate(axis=-1)([
        resnet_front,
        resnet_rear,
        resnet_passenger_side,
        resnet_driver_side
    ])

    # Each of our filters doubles the size of the previous layer due to the
    # Upsampling layer - so 8->16->32->64->128->256
    for filters in [512, 256, 128, 64, 32]:
        # net = LeakyReLU(net)
        net = Conv2DTranspose(filters, 3, padding="same",
                              activation="leaky_relu")(net)
        net = BatchNormalization()(net)

        net = Conv2DTranspose(filters, 3, padding="same",
                              activation="leaky_relu")(net)
        net = BatchNormalization()(net)

        net = UpSampling2D(2)(net)

    # 3 filters for 3 channels of our output
    outputs = Conv2DTranspose(3, 3, padding="same", activation="tanh")(net)

    model = keras.Model(inputs, outputs)

    return model
