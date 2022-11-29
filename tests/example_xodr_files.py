import os
from typing import Dict, List

import pytest as pt
from rich.progress import track

from pyxodr.road_objects.network import RoadNetwork

open_drive_example_directory = os.path.join("tests", "example_networks")

if not os.path.isdir(open_drive_example_directory):
    raise ValueError(
        f"OpenDRIVE example directory {os.path.abspath(open_drive_example_directory)} "
        + "does not exist."
    )

# Some OpenDRIVE files are expected to fail, due to limitations described in the README.
# We remove them here, and they should be added back as functionality is added.
# Removed files:
# Ex_LHT-Complex-X-Junction
#     - My understanding is that there is an error in this file in the 1.7.0 spec, see
#       note in README
# UC_ParamPoly3
#     - Since this file uses elementS / elementDir for road and junction linking, which
#       is not supported (and noted in the TODO in the README.)
filenames_to_remove = {"Ex_LHT-Complex-X-Junction", "UC_ParamPoly3"}

example_xodr_file_paths: List[str] = []
for subdir, dirs, files in os.walk(open_drive_example_directory):
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".xodr" and filename not in filenames_to_remove:
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
