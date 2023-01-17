import sys

from birdseye.encoder.model import Decoder, Discriminator, Encoder
from birdseye.encoder.train import (
    train_birdseye_encoder,
    train_birdseye_encoder_with_discriminator,
)
from birdseye.naive.model import BirdsEyeNaive
from birdseye.naive.train import train_birdseye_naive


def train_naive():
    model = BirdsEyeNaive()
    model = train_birdseye_naive(model, 50, 8, "./dataset")
    model.save("./finished_model")


def train_encoder_decoder():
    encoder = Encoder()
    decoder = Decoder()
    model = train_birdseye_encoder(
        encoder, decoder, "front", 50, 16, "./dataset", 0.0,
        "./checkpoints_encoder")
    model.save("./encoder_model")


def train_enc_dec_disc():
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    model = train_birdseye_encoder_with_discriminator(
        encoder,
        decoder,
        discriminator,
        "front",
        50,
        16,
        "./dataset"
    )
    model.save("./encoder_w_disc_model")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please provide which network you wish to train")
        sys.exit(0)

    funcs = {
        "naive": train_naive,
        "encoder_decoder": train_encoder_decoder,
        "enc_dec_disc": train_enc_dec_disc
    }

    if sys.argv[1] not in funcs:
        print(f"I'm afraid I'm not sure what you mean by {sys.argv[1]}")
        sys.exit(0)

    funcs[sys.argv[1]]()
