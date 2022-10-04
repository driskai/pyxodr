import math
from typing import Tuple

# Based on
# https://github.com/fcagroupj/openDrive/blob/master/OdrSpiral.py
# or (particularly for comments / documentation)
# https://github.com/DLR-TS/odrSpiral/blob/master/src/odrSpiral.c

# Local hardcoded variables for function parameters
# S(x) for small x
sn = [
    -2.99181919401019853726e3,
    7.08840045257738576863e5,
    -6.29741486205862506537e7,
    2.54890880573376359104e9,
    -4.42979518059697779103e10,
    3.18016297876567817986e11,
]
sd = [
    2.81376268889994315696e2,
    4.55847810806532581675e4,
    5.17343888770096400730e6,
    4.19320245898111231129e8,
    2.24411795645340920940e10,
    6.07366389490084639049e11,
]

# C(x) for small x
cn = [
    -4.98843114573573548651e-8,
    9.50428062829859605134e-6,
    -6.45191435683965050962e-4,
    1.88843319396703850064e-2,
    -2.05525900955013891793e-1,
    9.99999999999999998822e-1,
]
cd = [
    3.99982968972495980367e-12,
    9.15439215774657478799e-10,
    1.25001862479598821474e-7,
    1.22262789024179030997e-5,
    8.68029542941784300606e-4,
    4.12142090722199792936e-2,
    1.00000000000000000118e0,
]

# Auxiliary function f(x)
fn = [
    4.21543555043677546506e-1,
    1.43407919780758885261e-1,
    1.15220955073585758835e-2,
    3.45017939782574027900e-4,
    4.63613749287867322088e-6,
    3.05568983790257605827e-8,
    1.02304514164907233465e-10,
    1.72010743268161828879e-13,
    1.34283276233062758925e-16,
    3.76329711269987889006e-20,
]
fd = [
    7.51586398353378947175e-1,
    1.16888925859191382142e-1,
    6.44051526508858611005e-3,
    1.55934409164153020873e-4,
    1.84627567348930545870e-6,
    1.12699224763999035261e-8,
    3.60140029589371370404e-11,
    5.88754533621578410010e-14,
    4.52001434074129701496e-17,
    1.25443237090011264384e-20,
]

# Auxiliary function g(x)
gn = [
    5.04442073643383265887e-1,
    1.97102833525523411709e-1,
    1.87648584092575249293e-2,
    6.84079380915393090172e-4,
    1.15138826111884280931e-5,
    9.82852443688422223854e-8,
    4.45344415861750144738e-10,
    1.08268041139020870318e-12,
    1.37555460633261799868e-15,
    8.36354435630677421531e-19,
    1.86958710162783235106e-22,
]
gd = [
    1.47495759925128324529e0,
    3.37748989120019970451e-1,
    2.53603741420338795122e-2,
    8.14679107184306179049e-4,
    1.27545075667729118702e-5,
    1.04314589657571990585e-7,
    4.60680728146520428211e-10,
    1.10273215066240270757e-12,
    1.38796531259578871258e-15,
    8.39158816283118707363e-19,
    1.86958710162783236342e-22,
]


class OdrSpiral:
    """
    Class representing a standard Euler spiral, starting with a curvature of 0.

    Parameters
    ----------
    curvature_rate_of_change : float
        First derivative of curvature (1/m2)
    """

    def __init__(self, curvature_rate_of_change: float):
        self.curvature_rate_of_change = curvature_rate_of_change

    def _polevl(self, x, coef, n):
        ans = coef[0]
        for ii in range(n):
            ans = ans * x + coef[ii + 1]
        return ans

    def _p1evl(self, x, coef, n):
        ans = x + coef[0]
        for ii in range(1, n):
            ans = ans * x + coef[ii]
        return ans

    def _fresnel(self, xxa):
        x = math.fabs(xxa)
        x2 = x * x
        if x2 < 2.5625:
            t = x2 * x2
            ss = x * x2 * self._polevl(t, sn, 5) / self._p1evl(t, sd, 6)
            cc = x * self._polevl(t, cn, 5) / self._polevl(t, cd, 6)
        elif x > 36974.0:
            cc = 0.5
            s = 0.5
        else:
            x2 = x * x
            t = math.pi * x2
            u = 1.0 / (t * t)
            t = 1.0 / t
            f = 1.0 - u * self._polevl(u, fn, 9) / self._p1evl(u, fd, 10)
            g = t * self._polevl(u, gn, 10) / self._p1evl(u, gd, 11)

            t = math.pi * 0.5 * x2
            c = math.cos(t)
            s = math.sin(t)
            t = math.pi * x
            cc = 0.5 + (f * s - g * c) / t
            ss = 0.5 - (f * c + g * s) / t

        if xxa < 0.0:
            cc = -cc
            ss = -ss

        return (ss, cc)

    def __call__(self, s: float) -> Tuple[float, float, float]:
        """
        Compute the (x, y, t) coordinate of the spiral for length along spiral s.

        Parameters
        ----------
        s : float
            Run-length along the spiral.

        Returns
        -------
        Tuple[float, float, float]
            (
                resulting x coord in spiral's local coordinate system [m],
                resulting y coord in spiral's local coordinate system [m],
                tangent direction at s [rad]
            )
        """
        a = 1.0 / math.sqrt(math.fabs(self.curvature_rate_of_change))
        a *= math.sqrt(math.pi)
        (y, x) = self._fresnel(s / a)
        y *= a
        x *= a
        if self.curvature_rate_of_change < 0.0:
            y *= -1.0

        t = s * s * self.curvature_rate_of_change * 0.5

        return (x, y, t)
