import functools
from typing import Dict, List, Set

import numpy as np
from lxml import etree

from pyxodr.road_objects.road import Road


class Junction:
    """
    Class representing a Junction in an OpenDRIVE file.

    Parameters
    ----------
    junction_xml : etree._Element
        XML element corresponding to this junction.
    """

    def __init__(
        self,
        junction_xml: etree._Element,
    ):
        self.junction_xml = junction_xml

    @functools.cached_property
    def _connection_attributes_list(self) -> List[dict]:
        __connection_attributes = []
        for connection_xml in self.junction_xml.findall("connection"):
            __connection_attributes.append(connection_xml.attrib)
        return __connection_attributes

    def get_connecting_road_ids(self) -> Set[int]:
        """
        Return a set of ids of the connecting roads in this junction.

        See 10.3 in the OpenDRIVE spec.

        Returns
        -------
        Set[int]
            Set of int road ids making up the connecting roads in this junction.
        """
        _connecting_road_ids = set()
        for connection_attributes in self._connection_attributes_list:
            try:
                connecting_road_id = int(connection_attributes["connectingRoad"])
                _connecting_road_ids.add(connecting_road_id)
            except KeyError:
                pass

        return _connecting_road_ids

    def get_linked_road_ids(self) -> Set[int]:
        """
        Return a set of ids of the linked roads in this (direct) junction.

        See 10.4 in the OpenDRIVE spec.

        Returns
        -------
        Set[int]
            Set of int road ids making up the linked roads in this junction.
        """
        _linked_road_ids = set()
        for connection_attributes in self._connection_attributes_list:
            try:
                linked_road_id = int(connection_attributes["linkedRoad"])
                _linked_road_ids.add(linked_road_id)
            except KeyError:
                pass

        return _linked_road_ids

    def get_incoming_road_ids(self) -> Set[int]:
        """
        Return a set of ids of the incoming roads to this junction.

        See 10.2 in the OpenDRIVE spec.

        Returns
        -------
        Set[int]
            Set of int road ids making up the incoming roads to this junction.
        """
        _incoming_road_ids = set()
        for connection_attributes in self._connection_attributes_list:
            connecting_road_id = int(connection_attributes["incomingRoad"])
            _incoming_road_ids.add(connecting_road_id)
        return _incoming_road_ids

    def get_outgoing_road_ids(
        self, road_ids_to_objects: Dict[int, Road], fail_on_key_error: bool = True
    ) -> Set[int]:
        """
        Return a set of ids of the outgoing roads from this junction.

        Note OpenDRIVE doesn't specifically define outgoing roads. Therefore we have
        to get these roads from the successor ids of the connecting roads, hence a
        different method signature.

        Parameters
        ----------
        road_ids_to_objects : Dict[int, Road]
            Dictionary linking road ids to Road objects.
        fail_on_key_error : bool, optional
            If True, connecting road ids from this junction not present in the
            road_ids_to_objects dict keys will raise a KeyError, by default True

        Returns
        -------
        Set[int]
            A set of int roads ids making up the outgoing roads from this junction.

        Raises
        ------
        KeyError
            If fail_on_key_error and a connecting road id is found in this junction
            that is not present in the road_ids_to_objects dict keys.
        """
        _outgoing_road_ids = set()
        for connecting_road_id in self.get_connecting_road_ids():
            try:
                _outgoing_road_ids |= road_ids_to_objects[
                    connecting_road_id
                ].successor_ids
            except KeyError as ke:
                if fail_on_key_error:
                    raise ke
                else:
                    print(ke)

        # Also add linked roads (for direct junctions)
        _outgoing_road_ids |= self.get_linked_road_ids()

        return _outgoing_road_ids

    def __getitem__(self, name):
        return self.junction_xml.attrib[name]

    @property
    def id(self):
        """Get the OpenDRIVE ID of this junction."""
        return int(self["id"])

    def closest_point_on_road(self, road_obj: Road) -> np.ndarray:
        """
        Find the closest coordinate in the reference line of road_obj.

        Parameters
        ----------
        road_obj : Road
            Road object to return the closest reference line coordinate from.

        Returns
        -------
        np.ndarray
            Closest reference line coordinate.

        Raises
        ------
        ValueError
            If the provided road doesn't connect to this junction.
        """
        closest_point = None
        for (
            position,
            connecting_junction_ids,
        ) in road_obj.junction_connecting_ids.items():
            if self.id in connecting_junction_ids:
                if position == "predecessor":
                    closest_point = road_obj.reference_line[0]
                else:
                    closest_point = road_obj.reference_line[-1]
        if closest_point is None:
            raise ValueError(
                f"Road {road_obj.id} doesn't seem to connect to junction {self.id}"
            )
        return closest_point
