from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from pyxodr.geometries.base import Geometry


class CubicPolynom(Geometry):
    r"""
    Class representing the Cubic polynom from the OpenDRIVE spec (depreciated).

    $$y(x) = a + b*x + c*x2 + d*x^3$$

    Parameters
    ----------
    a : float
        a parameter in the interpolation equation.
    b : float
        b parameter in the interpolation equation.
    c : float
        c parameter in the interpolation equation.
    d : float
        d parameter in the interpolation equation.
    length : float, optional
        Length of the cubic polynomial line, by default None (in which case the call
        method of this class will be unusable)
    """

    def __init__(
        self, a: float, b: float, c: float, d: float, length: Optional[float] = None
    ):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.length = length

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
        if self.length is None:
            raise NotImplementedError(
                "Unable to parametrically call CubicPolynom when length is None"
            )
        else:
            u_array = p_array * self.length
            return self.u_v_from_u(u_array)

    def _du_ds_differential_equation(self, _: float, y: np.ndarray) -> np.ndarray:
        r"""
        Differential equation for use in the solving of u from s.

        Solves the equation
        $$\frac{du}{ds} = \frac{1}{\sqrt{1 + \left ( \frac{dv}{du} \right ) }}$$
        Where for this class
        $$v(u) = a + bu + cu^2 + du^3$$

        Parameters
        ----------
        _ : float
            Present to fit the function signature required by scipy's solve_ivp
        y : np.ndarray
            Array of K 1D y values, [1, K]. Note y here refers to the
            scipy.integrate.solve_ivp spec; for our use case, y == u.

        Returns
        -------
        np.ndarray
            Array of dy /dt (for our purposes, du / ds) values, [1, K]
        """
        # y : [1, K]
        y = y.T.squeeze()
        # y : [K]
        dv_du = np.ones_like(y) * self.b + 2 * self.c * y + 3 * self.d * np.power(y, 2)
        result = np.power(np.ones_like(y) + np.power(dv_du, 2), -0.5)
        return result.T

    def _u_array_from_arc_lengths(self, s_array: np.ndarray) -> np.ndarray:
        r"""
        Return an array of u (local coord) from s (distance along geometry).

        Required as OpenDRIVE provides a length value for road reference lines (i.e.
        a max s value) but the cubic polynomial geometry is parameterised by u (see
        7.6.2). Converting between them is done here by solving the initial value
        problem of the equation
        $$\frac{du}{ds} = \frac{1}{\sqrt{1 + \left ( \frac{dv}{du} \right ) }}$$
        with
        $$s_0 = 0$$
        $$u(s_0) = 0$$
        This seems convoluted so it's possible I've misunderstood the spec here;
        please raise an issue if so.

        Parameters
        ----------
        s_array : np.ndarray
            Array of distances along the polynomial curve.

        Returns
        -------
        np.ndarray
            Array of u values corresponding to these distances.
        """
        assert min(s_array) == 0.0
        solution = solve_ivp(
            self._du_ds_differential_equation,
            (0.0, max(s_array)),
            np.array([0.0]),
            t_eval=s_array,
            vectorized=True,
        )
        assert (s_array == solution.t).all()
        u_array = solution.y.squeeze()
        return u_array

    def u_v_from_arc_length(self, s_array: np.ndarray) -> np.ndarray:
        """
        Return local (u, v) coordinates from an array of s values (arc lengths).

        (u, v) coordinates are in their own x,y frame: start at origin, and initial
        heading is along the x axis.

        Parameters
        ----------
        s_array : np.ndarray
            s values corresponding to length along cubic polynomial line.

        Returns
        -------
        np.ndarray
            Array of local (u, v) coordinate pairs.
        """
        u_array = self._u_array_from_arc_lengths(s_array)
        local_coords = self.u_v_from_u(u_array)
        return local_coords

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
        v_array = (
            self.a * np.ones_like(u_array)
            + self.b * u_array
            + self.c * np.power(u_array, 2)
            + self.d * np.power(u_array, 3)
        )

        local_coords = np.stack((u_array, v_array), axis=1)

        return local_coords


class ParamCubicPolynom(Geometry):
    """
    Class representing the parametric cubic curve from the OpenDRIVE spec.

    Parameters
    ----------
    aU : float
        a parameter for the U curve.
    bU : float
        b parameter for the U curve.
    cU : float
        c parameter for the U curve.
    dU : float
        d parameter for the U curve.
    aV : float
        a parameter for the V curve.
    bV : float
        b parameter for the V curve.
    cV : float
        c parameter for the V curve.
    dV : float
        d parameter for the V curve.
    """

    def __init__(
        self,
        aU: float,
        bU: float,
        cU: float,
        dU: float,
        aV: float,
        bV: float,
        cV: float,
        dV: float,
    ):
        # From the OpenDRIVE spec (7.7.1) we can use the call function from the
        # CubicPolynom class above but parameterised over the range [0,1]. We can
        # achieve this by setting the length of these curves to 1.0.
        self.pU = CubicPolynom(aU, bU, cU, dU, length=1.0)
        self.pV = CubicPolynom(aV, bV, cV, dV, length=1.0)

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
        u_array = self.pU(p_array)[:, 1]
        v_array = self.pV(p_array)[:, 1]

        local_coords = np.stack((u_array, v_array), axis=1)
        return local_coords

    def u_v_from_u(self, u_array: np.ndarray) -> np.ndarray:
        """Raise an error; this geometry is parameteric with no v from u method."""
        raise NotImplementedError("This geometry is only defined parametrically.")
