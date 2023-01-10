from birdseye.naive.model import BirdsEye
from birdseye.naive.train import train_birdseye_naive

model = BirdsEye()
model = train_birdseye_naive(model, 50, 8, "./dataset")

model.save("./finished_model")
