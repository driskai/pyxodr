"""Test that the lane lines appear to be drivable in every loaded network."""

import os
from typing import Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pytest as pt

from pyxodr.road_objects.network import RoadNetwork
from tests.example_xodr_files import example_xodr_file_paths

lane_types_to_be_ignored = [None, set(["sidewalk", "shoulder"]), set(["driving"])]


@pt.mark.parametrize("xodr_path", example_xodr_file_paths)
@pt.mark.parametrize("ignored_lane_types", lane_types_to_be_ignored)
def test_plot(xodr_path: str, ignored_lane_types: Optional[Set[str]]):
    """Test that the road network plots without errors."""
    rn = RoadNetwork(xodr_path, ignored_lane_types=ignored_lane_types)
    road_network_name = os.path.basename(xodr_path).split(".")[0]

    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    rn.plot(ax, plot_start_and_end=True)
    plt.savefig(
        os.path.join(
            "tests",
            "output_plots",
            f"{road_network_name}_ignored_{ignored_lane_types}.pdf",
        )
    )
    plt.close()


@pt.mark.parametrize("xodr_path", example_xodr_file_paths)
@pt.mark.parametrize("ignored_lane_types", lane_types_to_be_ignored)
def test_no_right_angles(xodr_path: str, ignored_lane_types: Optional[Set[str]]):
    """Test that the cosine similarity of successive direction vectors is never < 0."""
    rn = RoadNetwork(xodr_path, ignored_lane_types=ignored_lane_types)
    road_network_name = os.path.basename(xodr_path).split(".")[0]
    for road in rn.get_roads(include_connecting_roads=True):
        for lane_section in road.lane_sections:
            for lane in lane_section.lanes:
                lane_traffic_flow_line = lane.traffic_flow_line
                lane_diffs = lane_traffic_flow_line[1:] - lane_traffic_flow_line[:-1]
                lane_unit_diffs = (lane_diffs.T / np.linalg.norm(lane_diffs, axis=1)).T

                lane_dot_products = np.einsum(
                    "ij,ij->i", lane_unit_diffs[:-1], lane_unit_diffs[1:]
                )

                minimum_lane_similarity = lane_dot_products.min()

                if minimum_lane_similarity <= 0.0:
                    plt.figure(figsize=(20, 20))
                    ax = plt.gca()
                    rn.plot(ax)
                    min_similarity_index = np.argmin(lane_dot_products)
                    min_similarity_xy_coord = np.mean(
                        lane_traffic_flow_line[
                            min_similarity_index : min_similarity_index + 1
                        ][:, :2],
                        axis=0,
                    )
                    min_similarity_radius = np.linalg.norm(
                        lane_traffic_flow_line[min_similarity_index + 1]
                        - lane_traffic_flow_line[min_similarity_index]
                    )
                    similarity_circle = plt.Circle(
                        min_similarity_xy_coord,
                        max(1.0, 2 * min_similarity_radius),
                        fill=False,
                        color="r",
                    )
                    ax.add_patch(similarity_circle)
                    plt.savefig(
                        os.path.join(
                            "tests",
                            "output_plots",
                            f"{road_network_name}_right_angle.pdf",
                        )
                    )
                    raise ValueError(
                        f"Minimum cosine similarity of lane {lane} is "
                        + f" {minimum_lane_similarity}; less than 0."
                    )

                # Also check lane successors
                for successor_lane in lane.traffic_flow_successors:
                    source_end_of_lane = lane_traffic_flow_line[-1]
                    # Skip 0 as they may be on top of each other
                    destination_start_of_lane = successor_lane.traffic_flow_line[1]

                    lane_successor_difference_vector = (
                        destination_start_of_lane - source_end_of_lane
                    )
                    unit_lane_successor_difference_vector = (
                        lane_successor_difference_vector
                        / np.linalg.norm(lane_successor_difference_vector)
                    )

                    source_cosine_similarity = np.dot(
                        unit_lane_successor_difference_vector, lane_unit_diffs[-1]
                    )

                    destination_lane_start_diff = (
                        successor_lane.traffic_flow_line[1]
                        - successor_lane.traffic_flow_line[0]
                    )
                    destination_lane_start_unit_diff = (
                        destination_lane_start_diff
                        / np.linalg.norm(destination_lane_start_diff)
                    )

                    destination_cosine_similarity = np.dot(
                        unit_lane_successor_difference_vector,
                        destination_lane_start_unit_diff,
                    )

                    if (
                        source_cosine_similarity < 0.0
                        or destination_cosine_similarity < 0.0
                    ):
                        plt.figure(figsize=(20, 20))
                        ax = plt.gca()
                        rn.plot(ax)
                        similarity_circle = plt.Circle(
                            np.mean(
                                np.array(
                                    [source_end_of_lane, destination_start_of_lane]
                                ),
                                axis=0,
                            ),
                            max(
                                1.0,
                                0.75
                                * np.linalg.norm(
                                    destination_start_of_lane - source_end_of_lane
                                ),
                            ),
                            fill=False,
                            color="r",
                        )
                        ax.add_patch(similarity_circle)
                        plt.savefig(
                            os.path.join(
                                "tests",
                                "output_plots",
                                f"{road_network_name}_"
                                + f"ignored_{ignored_lane_types}_right_angle.pdf",
                            )
                        )
                        if source_cosine_similarity < 0.0:
                            raise ValueError(
                                "Cosine similarity between final direction vector of "
                                + f"lane {lane} and lane change vector to successor "
                                + f"lane {successor_lane} is "
                                + f"{source_cosine_similarity}."
                            )
                        else:
                            raise ValueError(
                                "Cosine similarity between initial direction vector of "
                                + f"lane {successor_lane} and lane change vector from "
                                + f"predecessor lane {lane} is "
                                + f"{destination_cosine_similarity}."
                            )
