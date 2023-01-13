from keras.applications import ResNet50
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    UpSampling2D,
    add,
)
from tensorflow import keras


def Encoder():
    """
    Encoder returns a singular model that is setup to take in an image
    and then a decoder undoes it.
    """
    input_shape = (256, 256, 3)
    inputs = keras.Input(shape=input_shape)

    outputs = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name="encoder")
    return model


def Decoder():
    """
    This takes the output of the Encoder and attempts to recreate the
    input.
    """
    # The output shape of the ResNet50 top layers is(batch, 8, 8, 2048)
    input_shape = (8, 8, 2048)
    inputs = keras.Input(shape=input_shape)

    net = Conv2D(1024, 3, padding="same")(inputs)
    net = BatchNormalization()(net)
    net = Activation("leaky_relu")(net)

    skip_memory = net

    # Let's build up and outwards
    for filters in [512, 256, 128, 64, 32]:
        net = Conv2DTranspose(filters, 3, padding="same",
                              activation="leaky_relu")(net)
        net = BatchNormalization()(net)
        net = Conv2DTranspose(filters, 3, padding="same",
                              activation="leaky_relu")(net)
        net = BatchNormalization()(net)
        net = UpSampling2D(2)(net)

        # Handle skip layer
        skip_memory = UpSampling2D(2)(skip_memory)
        skip_memory = Conv2D(filters, 1, padding="same")(skip_memory)
        net = add([net, skip_memory])
        skip_memory = net

    # 3 filters for 3 channels of our output
    outputs = Conv2DTranspose(3, 3, padding="same", activation="tanh")(net)

    model = keras.Model(inputs=inputs, outputs=outputs, name="decoder")

    return model
