from __future__ import annotations

from enum import Enum
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lxml import etree

from pyxodr.geometries import CubicPolynom, MultiGeom
from pyxodr.utils import cached_property


class LaneOrientation(Enum):
    """Enum representing whether a lane is left or right of the reference line."""

    LEFT = 0
    RIGHT = 1


class TrafficOrientation(Enum):
    """
    Enum representing whether a road is right-hand-drive or left-hand-drive.

    If right hand drive, lanes with positive IDs will have their centre line directions
    flipped (as these lanes will be on the left of the road reference line
    and should be going in the opposite direciton to it).
    """

    RIGHT = 0
    LEFT = 1


class ConnectionPosition(Enum):
    """Whether a connection connects to the beginning or end of a specified object."""

    BEGINNING = 0
    END = -1

    @classmethod
    def from_contact_point_str(cls, contact_point_str: str) -> ConnectionPosition:
        """
        Create a ConnectionPosition enum from an OpenDRIVE e_contactPoint string.

        See OpenDRIVE Spec Table 105.

        Parameters
        ----------
        contact_point_str : str
            e_contactPoint string.

        Returns
        -------
        ConnectionPosition
            Connection Position enum.

        Raises
        ------
        ValueError
            If an unknown contact point string is passed in.
        """
        if contact_point_str == "end":
            return cls.END
        elif contact_point_str == "start":
            return cls.BEGINNING
        else:
            raise ValueError(
                f"Unknown contact point str {contact_point_str}: "
                + "expected 'end' or 'start'."
            )

    @property
    def index(self):
        """Get the numerical index, to be used to index a list."""
        return self.value


class Lane:
    """
    Class representing a Lane in an OpenDRIVE file.

    Parameters
    ----------
    road_id : int
        Parent road ID.
    lane_section_id : int
        Parent lane section ID (ordinal - see lane_section_ordinal description in
        the LaneSection class docstring).
    lane_xml : etree._Element
        XML element corresponding to this lane.
    lane_offset_line : np.ndarray
        Offset line of this lane. This will be a sub-section of the lane offset line
        (OpenDRIVE spec 9.3) that just covers the parent lane section, rather than
        the whole road.
    lane_section_reference_line : np.ndarray
        Reference line of this lane. This will be a sub-section of the road
        reference line (OpenDRIVE spec 7.1) that just covers the parent lane
        section, rather than the whole road.
    orientation : LaneOrientation
        Enum representing if this lane is on the left or the right of the reference
        line.
    traffic_orientation: TrafficOrientation
        The traffic orientation (right/left-hand-drive) for this lane. See
        OpenDRIVE Spec Section 8.
    lane_z_coords : np.ndarray
        z coordinates of the reference line of this lane. This will be the z coordinates
        of a sub-section of the road reference line (OpenDRIVE spec 7.1) that just
        covers the parent lane section, rather than the whole road.
    inner_lane : Lane, optional
        The Lane on the inside edge of this lane (one further to the road reference
        line), by default None
    """

    def __init__(
        self,
        road_id: int,
        lane_section_id: int,
        lane_xml: etree._Element,
        lane_offset_line: np.ndarray,
        lane_section_reference_line: np.ndarray,
        orientation: LaneOrientation,
        traffic_orientation: TrafficOrientation,
        lane_z_coords: np.ndarray,
        inner_lane: Lane = None,
    ):
        self.road_id = road_id
        self.lane_section_id = lane_section_id
        self.lane_xml = lane_xml
        self.orientation = orientation
        self.traffic_orientation = traffic_orientation
        self.lane_offset_line = lane_offset_line
        self.lane_section_reference_line = lane_section_reference_line
        self.lane_z_coords = lane_z_coords

        if inner_lane is None:
            self.lane_reference_line = lane_offset_line
        else:
            self.lane_reference_line = inner_lane.boundary_line

        self.successor_data: List[Tuple[Lane, str]] = []
        self.predecessor_data: List[Tuple[Lane, str]] = []

    def __getitem__(self, name):
        return self.lane_xml.attrib[name]

    def __hash__(self):
        return hash((self.road_id, self.lane_section_id, self.id))

    def __repr__(self):
        return f"Lane_{self.id}/Section_{self.lane_section_id}/Road_{self.road_id}"

    @property
    def id(self):
        """Get the OpenDRIVE ID of this lane."""
        return int(self["id"])

    @property
    def successor_ids(self) -> List[int]:
        """Get the OpenDRIVE IDs of the successor lanes to this lane."""
        link_xml = self.lane_xml.find("link")
        if link_xml is None:
            return []
        return [
            int(successor_xml.attrib["id"])
            for successor_xml in link_xml.findall("successor")
        ]

    @property
    def predecessor_ids(self) -> List[int]:
        """Get the OpenDRIVE IDs of the predecessor lanes to this lane."""
        link_xml = self.lane_xml.find("link")
        if link_xml is None:
            return []
        return [
            int(predecessor_xml.attrib["id"])
            for predecessor_xml in link_xml.findall("predecessor")
        ]

    @cached_property
    def type(self) -> str:
        """Get the OpenDRIVE type of this lane."""
        lane_type = self.lane_xml.attrib["type"]
        if lane_type == "none":
            lane_type = None
        return lane_type

    @cached_property
    def boundary_line(self) -> np.ndarray:
        """
        Return the boundary line of this lane.

        Note this is the _far_ boundary, i.e. furthest from the road centre

        Returns
        -------
        np.ndarray
            Boundary of the far edge of the lane.
        """
        if len(self.lane_section_reference_line) == 0:
            raise IndexError(f"Zero length reference line in lane {self}")

        lane_uses_widths = self.lane_xml.findall("width") != []
        lane_uses_borders = self.lane_xml.findall("border") != []

        if lane_uses_widths and lane_uses_borders:
            raise NotImplementedError(
                f"{self} seems to use both widths and borders; unsupported."
            )
        elif not lane_uses_widths and not lane_uses_borders:
            if self.type is None:
                return self.lane_section_reference_line
            raise NotImplementedError(
                f"{self} seems to use neither widths nor borders; unsupported "
                + "(for type!=none)."
            )

        lane_geometries = []
        lane_distances = []

        for element in self.lane_xml.findall("width" if lane_uses_widths else "border"):
            try:
                s = float(element.attrib["s"])
            except KeyError:
                s = float(element.attrib["sOffset"])
            a = float(element.attrib["a"])
            b = float(element.attrib["b"])
            c = float(element.attrib["c"])
            d = float(element.attrib["d"])

            lane_geometries.append(CubicPolynom(a, b, c, d))
            lane_distances.append(s)

        lane_multi_geometry = MultiGeom(lane_geometries, np.array(lane_distances))
        (
            global_lane_coords,
            _,
        ) = lane_multi_geometry.global_coords_and_offsets_from_reference_line(
            self.lane_section_reference_line,
            self.lane_offset_line,
            self.lane_reference_line if lane_uses_widths else self.lane_offset_line,
            direction="left" if self.orientation is LaneOrientation.LEFT else "right",
        )

        return global_lane_coords

    @cached_property
    def centre_line(self) -> np.ndarray:
        """
        Return the centre line of this lane.

        Centre line is computed as halfway between the lane reference line and the lane
        boundary.

        Returns
        -------
        np.ndarray
            Coordinates of the lane centre line.
        """
        lane_centre_xy = np.mean((self.lane_reference_line, self.boundary_line), axis=0)
        lane_centre = np.append(
            lane_centre_xy, self.lane_z_coords[:, np.newaxis], axis=1
        )
        return lane_centre

    @property
    def _traffic_flows_in_opposite_direction_to_centre_line(self) -> bool:
        """
        Return bool representing whether traffic flows in opposite direction to centre.

        Considering the traffic orientation (RHT or LHT) and whether this lane is to the
        right or to the left of the centre line.

        Returns
        -------
        bool
            True if the traffic flows in the opposite direction to the centre line.
        """
        # Negative ID means to the right of the road reference line.
        traffic_flows_in_opposite_direction_to_centre_line = (self.id < 0) != (
            self.traffic_orientation is TrafficOrientation.RIGHT
        )
        return traffic_flows_in_opposite_direction_to_centre_line

    @property
    def traffic_flow_line(self) -> np.ndarray:
        """
        Return the centre line in the direction along which traffic would flow.

        Returns
        -------
        np.ndarray
            Coordinates of the centre line in the order of (legal) traffic flow.
        """
        if self._traffic_flows_in_opposite_direction_to_centre_line:
            traffic_flow_line = np.flip(self.centre_line, axis=0)
        else:
            traffic_flow_line = self.centre_line

        return traffic_flow_line

    @property
    def traffic_flow_successors(self) -> Set[Lane]:
        """
        Return the successors of this lane that traffic could legally flow to.

        Returns
        -------
        Set[Lane]
            Set of successor lanes according to traffic flow.
        """
        successor_lanes = set([])
        if self._traffic_flows_in_opposite_direction_to_centre_line:
            successor_data = self.predecessor_data
        else:
            successor_data = self.successor_data

        for lane, connection_position in successor_data:
            if lane._traffic_flows_in_opposite_direction_to_centre_line:
                if connection_position is not ConnectionPosition.END:
                    raise ValueError(
                        f"Expected to connect to the end of {lane}, "
                        + "after flipping it according to traffic flow direction."
                    )
            else:
                if connection_position is not ConnectionPosition.BEGINNING:
                    raise ValueError(f"Expected to connect to the start of {lane}.")

            successor_lanes.add(lane)

        return successor_lanes

    def plot(
        self,
        axis: plt.Axes,
        plot_start_and_end: bool = False,
        line_scale_factor: float = 1.0,
    ) -> plt.Axes:
        """
        Plot a visualisation of lane on a provided axis object.

        Parameters
        ----------
        axis : plt.Axes
            Axis on which to plot the lane.
        plot_start_and_end : bool, optional
            If True, plot both the start and end of this lane (start with blue dot,
            end with pink cross), by default False
        line_scale_factor : float, optional
            Scale all lines thicknesses up by this factor, by default 1.0.

        Returns
        -------
        plt.Axes
            Axis with the lane plotted on it.
        """
        lane_traffic_flow_line = self.traffic_flow_line[:, :2]
        axis.plot(
            *lane_traffic_flow_line.T,
            "--",
            linewidth=0.2 * line_scale_factor,
            color="grey",
            alpha=0.8,
        )

        if plot_start_and_end:
            axis.scatter(
                [lane_traffic_flow_line[0][0]],
                [lane_traffic_flow_line[0][1]],
                marker="o",
                c="blue",
                s=4,
            )
            axis.scatter(
                [lane_traffic_flow_line[-1][0]],
                [lane_traffic_flow_line[-1][1]],
                marker="x",
                c="pink",
                s=4,
            )

        # Always plot lane directions
        origin_coordinate = lane_traffic_flow_line[len(lane_traffic_flow_line) // 2]
        try:
            arrow_difference_vector = (
                lane_traffic_flow_line[len(lane_traffic_flow_line) // 2 + 1]
                - origin_coordinate
            )
            axis.arrow(
                *origin_coordinate,
                *arrow_difference_vector,
                shape="full",
                lw=0.5,
                length_includes_head=True,
                head_width=0.5,
            )
        except IndexError as e:
            print(
                str(e)
                + " - this is likely caused by a lane which is too "
                + "short. A direction arrow will not be printed for "
                + f"{self}."
                + "\nIf you're seeing lots of these errors, try a "
                + "smaller (finer) resolution."
            )

        # And always plot lane connections
        for successor_lane in self.traffic_flow_successors:
            origin_coordinate = lane_traffic_flow_line[-1]
            # Skip 0 as they may be on top of each other
            arrow_difference_vector = (
                successor_lane.traffic_flow_line[1, :2] - origin_coordinate
            )
            axis.arrow(
                *origin_coordinate,
                *arrow_difference_vector,
                shape="full",
                lw=0.5,
                length_includes_head=True,
                head_width=0.5,
                color="red",
            )

        return axis
