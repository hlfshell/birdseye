from birdseye.model import BirdsEye
from birdseye.train import train_birdseye

model = BirdsEye()
model = train_birdseye(model, 50, 8, "./dataset")

model.save("./finished_model")
