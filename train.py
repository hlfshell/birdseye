import sys

from birdseye.encoder.model import Decoder, Encoder
from birdseye.encoder.train import train_birdseye_encoder
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


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Please provide which network you wish to train")
        sys.exit(0)

    funcs = {
        "naive": train_naive,
        "encoder_decoder": train_encoder_decoder
    }

    funcs[sys.argv[1]]()
