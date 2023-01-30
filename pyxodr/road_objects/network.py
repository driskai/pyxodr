from functools import lru_cache
from typing import List, Optional, Set

import matplotlib.pyplot as plt
from lxml import etree
from rich import print
from rich.progress import track

from pyxodr.road_objects.junction import Junction
from pyxodr.road_objects.lane import ConnectionPosition
from pyxodr.road_objects.road import Road
from pyxodr.utils import cached_property


class RoadNetwork:
    """
    Class representing a road network from an entire OpenDRIVE file.

    Parameters
    ----------
    xodr_file_path : str
        Filepath to the OpenDRIVE file.
    resolution : float, optional
        Spatial resolution (in m) with which to create the road object coordinates, by
        default 0.1
    ignored_lane_types : Set[str], optional
        A set of lane types that should not be read from the OpenDRIVE file. If
        unspecified, no types are ignored.
    """

    def __init__(
        self,
        xodr_file_path: str,
        resolution: float = 0.1,
        ignored_lane_types: Optional[Set[str]] = None,
    ):
        self.tree = etree.parse(xodr_file_path)
        self.root = self.tree.getroot()

        self.resolution = resolution

        self.ignored_lane_types = (
            set([]) if ignored_lane_types is None else ignored_lane_types
        )

        self.road_ids_to_object = {}

    @lru_cache(maxsize=None)
    def get_junctions(self) -> List[Junction]:
        """Return the Junction objects for all junctions in this road network."""
        junctions = []
        for junction_xml in self.root.findall("junction"):
            junctions.append(
                Junction(
                    junction_xml,
                )
            )
        return junctions

    @cached_property
    def connecting_road_ids(self) -> Set[int]:
        """Return the IDs of all connecting roads in all junctions in this network."""
        _connecting_road_ids = set()
        for junction in self.get_junctions():
            _connecting_road_ids |= junction.get_connecting_road_ids()
        return _connecting_road_ids

    def _link_roads(self):
        """
        Link all roads to their neighbours.

        Neighbours == successor and predecessor roads.
        Also kicks off a tree of method calls that connects all other road elements
        to their neighbours. The next method in the call tree is _link_lane_sections.
        """
        for road in self.road_ids_to_object.values():
            link_xmls = road.road_xml.findall("link")
            if link_xmls == []:
                continue
            elif len(link_xmls) > 1:
                raise ValueError("Expected roads to link to only one other road")
            else:
                link_xml = link_xmls[0]
            pred_xmls = link_xml.findall("predecessor")
            if pred_xmls == []:
                pred_xml = None
            elif len(pred_xmls) > 1:
                raise ValueError("Expected roads to have only one predecessor road.")
            else:
                pred_xml = pred_xmls[0]
            succ_xmls = link_xml.findall("successor")
            if succ_xmls == []:
                succ_xml = None
            elif len(succ_xmls) > 1:
                raise ValueError("Expected roads to have only one successor road.")
            else:
                succ_xml = succ_xmls[0]

            if pred_xml is not None:
                pred_dict = pred_xml.attrib
            else:
                pred_dict = None
            if succ_xml is not None:
                succ_dict = succ_xml.attrib
            else:
                succ_dict = None

            if pred_dict is not None and pred_dict["elementType"] == "road":
                road.predecessor_data = (
                    self.road_ids_to_object[pred_dict["elementId"]],
                    ConnectionPosition.from_contact_point_str(
                        pred_dict["contactPoint"]
                    ),
                )
            if succ_dict is not None and succ_dict["elementType"] == "road":
                road.successor_data = (
                    self.road_ids_to_object[succ_dict["elementId"]],
                    ConnectionPosition.from_contact_point_str(
                        succ_dict["contactPoint"]
                    ),
                )

            road._link_lane_sections()

    @lru_cache(maxsize=None)
    def get_roads(
        self,
        include_connecting_roads: bool = True,
        verbose: bool = False,
    ) -> List[Road]:
        """Return the Road objects for all roads in this network."""
        if not include_connecting_roads:
            ids_to_avoid = self.connecting_road_ids
        else:
            ids_to_avoid = set()
        roads = []
        iterator = self.root.findall("road")
        for road_xml in track(iterator) if verbose else iterator:
            road_id = road_xml.attrib["id"]
            if road_id in ids_to_avoid:
                continue
            if road_id in self.road_ids_to_object.keys():
                roads.append(self.road_ids_to_object[road_id])
            else:
                road = Road(
                    road_xml,
                    resolution=self.resolution,
                    ignored_lane_types=self.ignored_lane_types,
                )
                self.road_ids_to_object[road.id] = road
                roads.append(road)

        self._link_roads()

        return roads

    def plot(
        self,
        axis: plt.Axes,
        include_connecting_roads: bool = True,
        plot_junctions: bool = True,
        plot_lane_centres: bool = True,
        plot_start_and_end: bool = False,
        fail_on_key_error: bool = True,
        line_scale_factor: float = 1.0,
        label_size: Optional[int] = None,
    ) -> plt.Axes:
        """
        Plot a visualisation of this road network on a provided axis object.

        Parameters
        ----------
        axis : plt.Axes
            Axis on which to plot the road network.
        include_connecting_roads : bool, optional
            If True, also plot connecting roads, by default False
        plot_junctions : bool, optional
            If True, plot junction visualisations, by default True
        plot_lane_centres : bool, optional
            If True, plot the lane centres, by default True
        plot_start_and_end : bool, optional
            If True, plot both the start and end of roads and lanes (see their
            docstrings for colour details), by default False
        fail_on_key_error : bool, optional
            If True, connecting road ids from a junction not present in the
            road_ids_to_objects dict keys will raise a KeyError, by default True
        line_scale_factor : float, optional
            Scale all lines thicknesses up by this factor, by default 1.0.
        label_size : int, optional
            If specified, text of this font size will be displayed along each lane
            centre line of the form "l_n_s_m" where n is the ID of the lane, m is the id
            of the lane section, and along each road line of the form "r_n" where n is
            the ID of the road. By default None, resulting in no labels.

        Returns
        -------
        plt.Axes
            Axis with the road network plotted on it.

        Raises
        ------
        KeyError
            Where a connecting road id from a junction is not present in the
            road_ids_to_objects dict keys.
        """
        for road in self.get_roads(include_connecting_roads=include_connecting_roads):
            axis = road.plot(
                axis,
                plot_start_and_end=plot_start_and_end,
                line_scale_factor=line_scale_factor,
                label_size=label_size,
            )

            if plot_lane_centres:
                for lane_section in road.lane_sections:
                    for lane in lane_section.lanes:
                        axis = lane.plot(
                            axis,
                            plot_start_and_end=plot_start_and_end,
                            line_scale_factor=line_scale_factor,
                            label_size=label_size,
                        )

        # Visualise junctions
        if plot_junctions:
            for junction in self.get_junctions():
                closest_points = []
                for (
                    connected_road_id
                ) in junction.get_incoming_road_ids() | junction.get_outgoing_road_ids(
                    self.road_ids_to_object, fail_on_key_error=fail_on_key_error
                ):
                    try:
                        closest_points.append(
                            junction.closest_point_on_road(
                                self.road_ids_to_object[connected_road_id]
                            )
                        )
                    except KeyError as ke:
                        if fail_on_key_error:
                            raise ke
                        else:
                            print(ke)
                junction_centrepoint = sum(closest_points) / len(closest_points)
                axis.scatter(
                    [junction_centrepoint[0]],
                    [junction_centrepoint[1]],
                    s=2,
                    marker="o",
                    c="red",
                )
                for closest_point in closest_points:
                    diff = closest_point - junction_centrepoint
                    axis.arrow(
                        junction_centrepoint[0],
                        junction_centrepoint[1],
                        diff[0],
                        diff[1],
                    )
        return axis

    def plot_z(self, axis: plt.Axes, plot_lanes: bool = True):
        """
        Plot a 3D visualisation of the road network, with z coordinates.

        Parameters
        ----------
        axis : plt.Axes
            Axis on which to plot the network.
        plot_lanes : bool, optional
            If True, plot the lane centres, by default True

        Returns
        -------
        plt.Axes
            Axis with road network plotted.

        Raises
        ------
        IndexError
            If a NoneType lane has neighbours.
        """
        print("Plotting roads")
        for xodr_road in track(self.get_roads(include_connecting_roads=True)):
            for xodr_lane_section in xodr_road.lane_sections:
                x_centre, y_centre, z_centre = xodr_lane_section.get_offset_line().T

                axis.plot3D(x_centre, y_centre, z_centre, "gray")

                if plot_lanes:
                    for lane in xodr_lane_section.lanes:
                        lane_centre = lane.traffic_flow_line

                        x_lane, y_lane, z_lane = lane_centre.T
                        if lane.type is not None:
                            axis.plot3D(x_lane, y_lane, z_lane, "blue")
                        else:
                            if (
                                len(lane.successor_data) != 0
                                or len(lane.predecessor_data) != 0
                            ):
                                raise IndexError(
                                    "NoneType lanes are assumed to have no "
                                    + "connections. Found "
                                    + f"successors = {lane.successor_data} and "
                                    + f"predecessors = {lane.predecessor_data}"
                                )
        # To force real-world scaling
        # https://github.com/matplotlib/matplotlib/issues/17172
        # Not an ideal solution.
        axis.set_box_aspect(
            [ub - lb for lb, ub in (getattr(axis, f"get_{a}lim")() for a in "xyz")]
        )
        return axis
