import os
from typing import Dict

import pytest as pt
from rich.progress import track

from pyxodr.road_objects.network import RoadNetwork

open_drive_example_directory = os.environ.get("ODR_EXAMPLES")

if open_drive_example_directory is None:
    raise ValueError("Environment variable ODR_EXAMPLES is not defined.")
elif not os.path.isdir(open_drive_example_directory):
    raise ValueError(
        "Environment variable ODR_EXAMPLES is not a directory that exists."
    )

example_xodr_file_paths: list[str] = []
for subdir, dirs, files in os.walk(open_drive_example_directory):
    for file in files:
        _, file_extension = os.path.splitext(file)
        if file_extension == ".xodr":
            absolute_filepath = os.path.abspath(os.path.join(subdir, file))
            example_xodr_file_paths.append(absolute_filepath)


@pt.fixture(scope="module")
def loaded_road_networks() -> Dict[str, RoadNetwork]:
    """Load in all of the path strings to RoadNetwork objects."""
    print("Load in example road network files to road network objects.")
    road_network_to_loaded_network = {
        example_xodr_file_path: RoadNetwork(example_xodr_file_path)
        for example_xodr_file_path in track(
            example_xodr_file_paths, total=len(example_xodr_file_paths)
        )
    }
    return road_network_to_loaded_network
