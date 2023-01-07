from birdseye.carla.simulation import Simulation

from random import randrange

TIME_FOR_RUN_SECONDS = 60
NUMBER_OF_RUNS = 100
MIN_NUMBER_VEHICLES = 30
MAX_NUMBER_VEHICLES = 150
MIN_NUMBER_PEDESTRIANS = 20
MAX_NUMBER_PEDESTRIANS = 100

def single_run() -> str:
    number_of_vehicles = randrange(MIN_NUMBER_VEHICLES, MAX_NUMBER_VEHICLES + 1)
    number_of_pedestrians = randrange(MIN_NUMBER_PEDESTRIANS, MAX_NUMBER_PEDESTRIANS + 1)

    simulation = Simulation(vehicles=number_of_vehicles, pedestrians=number_of_pedestrians)
    simulation.launch(TIME_FOR_RUN_SECONDS)

    return simulation.run_id

def build_dataset(runs : int):
    for i in range(runs):
        id = single_run()
        print(f"\t-Completed run {i+1} : {id}")

if __name__ == '__main__':
    print("Running dataset collection")
    build_dataset(NUMBER_OF_RUNS)
    print(f"Completed running {NUMBER_OF_RUNS} runs...")