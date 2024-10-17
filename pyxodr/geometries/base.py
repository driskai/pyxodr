from abc import ABC
from typing import Optional
import numpy as np

from enum import Enum

class GeometryType(Enum):
    ARC = 0
    LINE = 1
    POLYNOMINAL = 2
    CUBIC_POLYNOM = 3
    PARAMETRIC_CUBIC_CURVE = 4
    SPIRAL = 5

class Geometry(ABC):
    """Base class for geometry objects."""
    def __init__(self, 
        geometry_type: GeometryType,
        x_offset: float,
        y_offset: float,
        heading_offset: float,
        length: Optional[float] = None
    ):
        self.geometry_type = geometry_type
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.heading_offset = heading_offset
        self.length = 1.0
        if length is not None:
            self.length = length

    @staticmethod
    def global_coords_from_offsets(
        local_coords: np.ndarray,
        x_offset: float,
        y_offset: float,
        heading_offset: float,
    ) -> np.ndarray:
        """
        Apply x and y (translation) and heading (rotation) offsets to local coords.

        Thereby generating coords in the global frame from coords in the local frame.
        N = number of coordinates
        D = dimension

        Parameters
        ----------
        local_coords : np.ndarray
            Array of local coords, [N, D].
        x_offset : float
            x offset value to be added to all local coordinates.
        y_offset : float
            y offset value to be added to all local coordinates.
        heading_offset : float
            Heading value (in radians) to rotate all local coordinates by.

        Returns
        -------
        np.ndarray
            Resultant coordinates in the global frame.
        """
        offset_coordinates = np.array([x_offset, y_offset])
        c, s = np.cos(heading_offset), np.sin(heading_offset)
        rotation_matrix = np.array(((c, -s), (s, c)))

        rotated_coords = np.dot(rotation_matrix, local_coords.T).T
        global_coords = rotated_coords + offset_coordinates

        return global_coords

    def evaluate_geometry(self, resolution: float):
        num_samples = max(int(self.length / resolution), 2)

        offsets = self(np.linspace(0.0, self.length, num_samples))
        return Geometry.global_coords_from_offsets(
                offsets,
                self.x_offset,
                self.y_offset,
                self.heading_offset,
            )
        
    @staticmethod
    def compute_offset_vectors(
        local_offsets: np.ndarray,
        reference_line_direction_vectors: np.ndarray,
        direction: str = "right",
    ) -> np.ndarray:
        """
        Compute offset vectors from line direction vectors & offset magnitudes.

        Consider a line made up of a series of xy coordinates. This method is for
        computing a line "alongside" it but offset from it.
        N = number of coordinates

        Parameters
        ----------
        local_offsets : np.ndarray
            Array of offset float values; these will be the magnitudes of the returned
            offset direction vectors, [N]
        reference_line_direction_vectors : np.ndarray
            Direction vectors of the reference line, [N, 2] (note this method works
            only in xy)
        direction : str, optional {"right", "left"}
            In which direction the offsets should point from the reference line,
            assuming facing in the direction of the reference line, by default "right"

        Returns
        -------
        np.ndarray
            Array of offset vectors. Add these to the original reference line
            coordinates to get the resultant offset coordinates, [N, 2]

        Raises
        ------
        ValueError
            If the specified direction is not supported.
        IndexError
            If the lengths of the local_offsets and reference_line_direction_vectors
            arrays are not equal.
        """
        if direction not in {"right", "left"}:
            raise ValueError("Unsupported direction, expected 'right' or 'left'")
        if len(local_offsets) != len(reference_line_direction_vectors):
            raise IndexError(
                f"Expected local offsets ({local_offsets.shape}) and reference line "
                + f"({reference_line_direction_vectors.shape}) to be of the same "
                + "length."
            )

        z_vector = np.array([0.0, 0.0, 1.0 if direction == "right" else -1.0])

        perpendicular_directions = np.cross(reference_line_direction_vectors, z_vector)
        # Reduce to 2D
        perpendicular_directions = perpendicular_directions[:, :-1]
        # replace any all-zero rows
        perpendicular_directions = fix_zero_directions(perpendicular_directions)
        # Unit scale
        scaled_perpendicular_directions_T = perpendicular_directions.T / np.linalg.norm(
            perpendicular_directions, axis=1
        )

        offsets = (local_offsets * scaled_perpendicular_directions_T).T

        return offsets

    @abstractmethod
    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        r"""
        Return local (u, v) coordinates from an array of parameter $p \in [0.0, 1.0]$.

        (u, v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along the x axis.

        Parameters
        ----------
        p_array : np.ndarray
            p values $\in [0.0, 1.0]$ to compute parametric coordinates.

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs.
        """
        ...

    @abstractmethod
    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """
        Return local (u, v) coordinates from an array of local u coordinates.

        (u, v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along the x axis.

        Parameters
        ----------
        u_array : np.ndarray
            u values from which to compute v values.

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs.
        """
        ...


class NullGeometry(Geometry):
    """Class for a "null geometry" which always returns zeros for local coords."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        r"""Return $(p, 0.0) \forall p \in p_array$."""
        v_array = np.zeros(len(p_array))

        local_coords = np.stack((p_array, v_array), axis=1)
        return local_coords

    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        r"""Return $(u, 0.0) \forall u \in u_array$."""
        v_array = np.zeros(len(u_array))

        local_coords = np.stack((u_array, v_array), axis=1)
        return local_coords
