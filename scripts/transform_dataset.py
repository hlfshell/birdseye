from pathlib import Path
from random import randint
from shutil import copy
import os

SOURCE_DIRECTORY = "./output"
TARGET_DIRECTORY = "./dataset"


def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)


def transform_dataset():
    Path(f"{TARGET_DIRECTORY}/input").mkdir(parents=True, exist_ok=True)
    Path(f"{TARGET_DIRECTORY}/overhead").mkdir(parents=True, exist_ok=True)
    Path(f"{TARGET_DIRECTORY}/semantic").mkdir(parents=True, exist_ok=True)

    directories = os.listdir(f"{SOURCE_DIRECTORY}/")

    for dir in directories:
        files = os.listdir(f"{SOURCE_DIRECTORY}/{dir}")

        # Group files so we collect them in sets of {frame#}_{type}.png
        images = {}
        for file in files:
            # Split the name into (frame, type)
            tmp = file.split("_")
            frame = tmp[0]
            # birdseye_semantic introduces another _, so deal with it here
            if len(tmp) == 3:
                tmp = "_".join(tmp[1:])
                tmp = tmp.split(".")
            else:
                tmp = tmp[1].split(".")

            image_type = tmp[0]

            if frame not in images:
                images[frame] = {}

            images[frame][image_type] = file

        # Now go over each collection and transfer it over with type dictating where
        for capture in images:
            short_id = random_with_N_digits(5)
            for image_type in images[capture].keys():
                file = images[capture][image_type]
                target_file = f"{SOURCE_DIRECTORY}/{dir}/{file}"
                if image_type == "birdseye":
                    copy(target_file,
                         f"{TARGET_DIRECTORY}/overhead/{short_id}_{file}")
                elif image_type == "birdseye_semantic":
                    copy(target_file,
                         f"{TARGET_DIRECTORY}/semantic/{short_id}_{file}")
                else:
                    copy(target_file,
                         f"{TARGET_DIRECTORY}/input/{short_id}_{file}")


if __name__ == "__main__":
    transform_dataset()
