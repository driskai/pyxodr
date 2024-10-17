import numpy as np

from pyxodr.geometries._standard_spiral import OdrSpiral
from pyxodr.geometries.base import Geometry


class Spiral(Geometry):
    """
    Class representing an Euler spiral / Clothoid geometry.

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
        curvStart: float,
        curvEnd: float,
        x_offset: float,
        y_offset: float,
        heading_offset: float,
        length: Optional[float] = None
    ):
        Geometry.__init__(self, GeometryType.SPIRAL, x_offset, y_offset, heading_offset, length)
        self.curvStart = curvStart
        self.curvEnd = curvEnd

        self.curvature_rate_of_change = (self.curvEnd - self.curvStart) / self.length

        self.standard_spiral = OdrSpiral(self.curvature_rate_of_change)

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
        s_offset = self.curvStart / self.curvature_rate_of_change
        offset_s_array = p_array * self.length + s_offset

        standard_coords = []
        for s in offset_s_array:
            standard_coords.append(self.standard_spiral(s))

        xy = np.array([(x, y) for x, y, _ in standard_coords])

        # Compute starting point of this standard spiral so we can translate it to the
        # correct point
        x0, y0, t0 = standard_coords[0]
        # Rotate
        angular_difference = t0
        xy_at_origin = xy - np.array([x0, y0])
        c, s = np.cos(angular_difference), np.sin(angular_difference)
        rm = np.array([[c, s], [-s, c]])
        # Apply rotation matrix:
        rotated_xy_at_origin = np.array(np.dot(rm, xy_at_origin.T).T)

        direction_vector = rotated_xy_at_origin[1] - rotated_xy_at_origin[0]
        direction_vector = direction_vector / np.linalg.norm(direction_vector)

        if np.linalg.norm(direction_vector - np.array([1.0, 0.0])) > 0.1:
            import matplotlib.pyplot as plt

            print(rotated_xy_at_origin)

            plt.figure(figsize=(20, 20))
            plt.plot(rotated_xy_at_origin.T[0], rotated_xy_at_origin.T[1])
            plt.savefig("spiral_error.pdf")
            raise ValueError(
                "Spiral seems to be going in wrong direction "
                + f"(vector {direction_vector})."
            )

        return rotated_xy_at_origin

    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """Raise an error; this geometry is parameteric with no v from u method."""
        raise NotImplementedError("This geometry is only defined parametrically.")
