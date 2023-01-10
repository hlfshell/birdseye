from keras.applications import ResNet50
from keras.layers import BatchNormalization, Concatenate, Conv2DTranspose, UpSampling2D
from tensorflow import keras


def Encoder():
    """
    Encoder returns a singular model that is setup to take in an image
    and then a decoder undoes it.
    """
    input_shape = (256, 256, 3)
    keras.Input(shape=input_shape)

    output = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    model = keras.Model(inputs=input, outputs=output)
    return model

def Decoder():
    """
    This takes the output of the Encoder and attempts to recreate the
    input.
    """
    # The output shape of the ResNet50 top layers is(batch, 8, 8, 2048)
    input_shape = (8, 8, 2048)
    keras.Input(shape=input_shape)

    # Let's build up and outwards
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

    model = keras.Model(inputs=input, outputs=outputs)

    return model