from copy import deepcopy
from typing import List

import numpy as np

from pyxodr.geometries.base import Geometry, NullGeometry


class MultiGeom:
    """
    Class representing sequential geometry objects.

    Parameters
    ----------
    geometries : List[Geometry]
        Ordered list of geometry objects.
    distance_array : np.ndarray
        Distances along reference line at which each of the geometries begin.
    """

    def __init__(self, geometries: List[Geometry], distance_array: np.ndarray):
        self.distance_array = distance_array
        self.geometries = geometries

        if len(self.geometries) != len(self.distance_array):
            raise IndexError("Geometry and distance arrays are of different lengths.")
        elif len(self.geometries) == 0:
            raise IndexError("Geometry and distance arrays are empty.")

    def __call__(self, u_array: np.ndarray) -> np.ndarray:
        """
        Return local (u, v) coordinates from a array of parameter u.

        (u,v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along x axis.
        Note that the u values will be translated to start from 0 in each case, but
        will not be scaled. This is to match use cases for e.g. z computation in the
        OpenDRIVE spec (elevationProfile)

        Parameters
        ----------
        u_array : np.ndarray
            Local u coordinates

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs
        """
        # Compute the index of the geometry to use for each s value
        # This is the last index where the distance value is less than the current s
        # value: equivalently one less than the first index where the s value is
        # exceeded
        geometry_indices = (
            np.argmax(
                np.tile(
                    np.concatenate((self.distance_array, np.array([np.inf]))),
                    (len(u_array), 1),
                ).T
                > u_array,
                axis=0,
            )
            - 1
        )

        du_values = u_array - self.distance_array[geometry_indices]

        v_values = []

        for geometry_index, geometry in enumerate(self.geometries):
            du_sub_values = du_values[geometry_indices == geometry_index]

            if len(du_sub_values) != 0:
                v_values.append(geometry.u_v_from_u(du_sub_values)[:, 1])

        local_coords = np.stack((u_array, np.concatenate(v_values)), axis=1)

        return local_coords

    def global_coords_and_offsets_from_reference_line(
        self,
        distance_line: np.ndarray,
        reference_line: np.ndarray,
        offset_line: np.ndarray,
        direction="right",
    ) -> np.ndarray:
        """
        Compute global coordinates of this multi geometry, given a reference line.

        Parameters
        ----------
        distance_line : np.ndarray
            The line of coords with which we will cross reference to determine where
            our various different geometries start and end (by comparing to our
            self.distance_array); i.e. our "s" / distance parameterisation
            corresponds to distance along this line.
            We also use this line to compute our offset directions (as perpendicular
            to the tangential direction vectors of this line).
        reference_line : np.ndarray
            The line to which we will actually add our offset vectors - in the
            OpenDRIVE spec this is useful for e.g. the Lane offset (9.3). Note it is
            distinct from the distance_line (above) as (s, t) are not defined relative
            to it, however it does itself represent an offset from the distance line.
        offset_line : np.ndarray
            The line representing an additional existing offset - e.g. in implementing
            lane boundaries in the OpenDRIVE spec, we do this by recursively calling
            lane boundaries from the inside out and using the previous lane's boundary
            along with the current lane's width (parameterised by s, which is itself
            computed along the distance line - see 9.5.1/calculation) to determine the
            current lane's boundary. This variable would be the previous lane's
            boundary.
        direction : str, optional {"right", "left"}
            In which direction the offsets should point from the reference line,
            assuming facing in the direction of the reference line, by default "right"

        Returns
        -------
        np.ndarray
            Global coordinates of the multi-geometry.
        """
        global_coordinates = []
        all_local_offsets = []

        # First, partition the reference line into subsections that fit into each
        # distance range
        distance_line_direction_vectors = np.diff(distance_line, axis=0)
        distance_line_distances = np.cumsum(
            np.linalg.norm(distance_line_direction_vectors, axis=1)
        )
        # Make the distances the same length as the original reference line
        # We add a 0 to the start as "the distance to the 0th element is 0"
        distance_line_distances = np.insert(distance_line_distances, 0, 0)
        # Repeat the final direction vector to give this the same shape as the centre
        # line
        # We repeat the final vector at the end as the best guess we have for the
        # direction at the final coordinate is the preceding direction.
        distance_line_direction_vectors = np.vstack(
            (distance_line_direction_vectors, distance_line_direction_vectors[-1])
        )
        # Make 3D for cross product
        distance_line_direction_vectors = np.vstack(
            (
                distance_line_direction_vectors.T,
                np.zeros(len(distance_line_direction_vectors)).T,
            )
        ).T
        s_values = self.distance_array.copy()
        partition_indices = np.searchsorted(distance_line_distances, s_values)

        existing_offsets = np.linalg.norm(offset_line - reference_line, axis=1)

        geometries = self.geometries

        # If we don't start from 0, we want to snap to the reference line until the
        # first index
        if partition_indices[0] != 0:
            partition_indices = np.insert(partition_indices, 0, 0)
            geometries = [NullGeometry()] + geometries
            s_values = np.insert(s_values, 0, 0.0)
        if partition_indices[-1] != len(reference_line):
            partition_indices = np.append(partition_indices, len(reference_line))
            geometries = geometries + [NullGeometry()]
            s_values = np.append(s_values, s_values[-1])

        start_end_indices = zip(partition_indices[:-1], partition_indices[1:])
        for (start_index, end_index), geometry, offset_distance in zip(
            start_end_indices, geometries, s_values
        ):
            if start_index != end_index:  # Ignore empty slices (e.g. at the end)
                sub_reference_line = reference_line[start_index:end_index]
                sub_reference_line_direction_vectors = distance_line_direction_vectors[
                    start_index:end_index
                ]
                sub_u_array = distance_line_distances[start_index:end_index]
                sub_global_offsets = existing_offsets[start_index:end_index]
                # Translate to start at 0
                sub_u_array -= offset_distance

                local_coords = (
                    geometry.u_v_from_u(sub_u_array)[:, 1] + sub_global_offsets
                )

                local_offsets = (
                    geometry.u_v_from_u(sub_u_array)[:, 1] + sub_global_offsets
                )
                all_local_offsets.append(local_offsets)

                if len(local_coords) != 0:
                    sub_global_coordinates = geometry.compute_offset_vectors(
                        local_coords,
                        sub_reference_line_direction_vectors,
                        direction=direction,
                    )
                    global_coordinates.append(
                        sub_reference_line + sub_global_coordinates
                    )
                else:
                    global_coordinates.append(deepcopy(sub_reference_line))

        global_coordinates = np.vstack(global_coordinates)
        all_local_offsets = np.concatenate(all_local_offsets)

        assert len(all_local_offsets) == len(reference_line)
        return global_coordinates, all_local_offsets
