import numpy as np

from pyxodr.geometries.base import Geometry


class Arc(Geometry):
    """
    Class representing an arc - a line with constant curvature.

    Parameters
    ----------
    curvature : float
        Curvature of the line [1/m]
    length : float
        Lenght of the arc [m]
    """

    def __init__(
        self,
        x_offset: float,
        y_offset: float,
        heading_offset: float,
        length: float,
        curvature: float
    ):
        Geometry.__init__(self, x_offset, y_offset, heading_offset, length)
        self.curvature = curvature

    def __call__(self, p_array: np.ndarray) -> np.ndarray:
        r"""
        Return local (p, v) coordinates from an array of parameter $p \in [0.0, 1.0]$.

        (p, v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along the x axis.

        Parameters
        ----------
        p_array : np.ndarray
            p values $\in [0.0, 1.0]$ to compute parametric coordinates.

        Returns
        -------
        np.ndarray
            Array of local (p, v) coordinate pairs.
        """
        # Just calculate circle segment
        radius_of_curvature = 1 / self.curvature
        circle_centre = np.array([0.0, radius_of_curvature])
        origin_coordinates_tensor = np.tile(
            np.array([circle_centre[0], circle_centre[1]]), (len(p_array), 1)
        )
        # From equation of arc length of circle and total length of arc
        max_theta = self.length / radius_of_curvature
        theta_array = p_array * max_theta

        u = radius_of_curvature * np.sin(theta_array)
        v = -radius_of_curvature * np.cos(theta_array)

        # Make sure to assert the first points go in the direction of the x axis
        direction_vector = np.array([u[1], v[1]]) - np.array([u[0], v[0]])
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        if np.linalg.norm(direction_vector - np.array([1.0, 0.0])) > 0.1:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(20, 20))
            plt.plot(u, v)
            plt.savefig("arc_error.pdf")
            raise ValueError(
                f"Arc seems to be going in wrong direction (vector {direction_vector})."
            )

        return np.array([u, v]).T + origin_coordinates_tensor

    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """Raise an error; this geometry is parameteric with no v from u method."""
        raise NotImplementedError("This geometry is only defined parametrically.")
