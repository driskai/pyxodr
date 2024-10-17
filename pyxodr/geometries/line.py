import numpy as np
from typing import Optional

from pyxodr.geometries.base import Geometry, GeometryType


class Line(Geometry):
    """
    Class representing an line geometry.

    Parameters
    ----------
    length : float
        Length [m] of the spiral.
    curvStart : float
        Curvature at the start of the spiral.
    curvEnd : float
        Curvature at the end of the spiral.
    """

    def __init__(
        self,
        x_offset: float,
        y_offset: float,
        heading_offset: float,
        length: Optional[float] = None
    ):
        Geometry.__init__(self, GeometryType.LINE, x_offset, y_offset, heading_offset, length)

    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        r"""
        Return local (u, v) coordinates from an array of parameter $p \in [0.0, 1.0]$.

        (u, v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along the x axis.
        In this method the algorithm to do this has steps:
        1) Compute the offset required to the p values to give us the correct starting
          curvature
        2) Compute the standard spiral
        3) Translate to origin and rotate so initial direction is along x axis

        Parameters
        ----------
        p_array : np.ndarray
            p values $\in [0.0, 1.0]$ to compute parametric coordinates.

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs.
        """
        
        # Construct direction vector
        direction_vector = np.array(
            [1.0, 0.0]
        )
        direction_tensor = np.tile(direction_vector, (len(p_array), 1))
        return (
            (direction_tensor.T * p_array).T
        )
        #geometry_coordinates.append(line_coordinates)

        #return np.array([(0.0, 0.0), (1.0, 0.0)])
    
    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """Raise an error; this geometry is parameteric with no v from u method."""
        raise NotImplementedError("This geometry is only defined parametrically.")

    def __str__(self) -> str:
        return ""
    