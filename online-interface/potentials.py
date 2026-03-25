"""
Shared potential definitions for app.py and worker.py.
All custom potentials use gala's `galactic` unit system (kpc, Myr, Msun, radian)
so parameter values are dimensionless numbers in those units.

Velocity note: 1 kpc/Myr ≈ 978 km/s  (so v_c=0.2 kpc/Myr ≈ 196 km/s)
"""

import gala.potential as gp
from gala.units import galactic

# ──────────────────────────────────────────────
# Parameter specs per potential
# Each entry: {"key": str, "label": str, "default": float, "min": float, "max": float}
# Potentials with no entries have no user-configurable parameters.
# ──────────────────────────────────────────────

POTENTIAL_PARAMS = {
    "MilkyWayPotential v2 (default)": [],
    "MilkyWayPotential v1":           [],
    "BovyMWPotential2014":            [],
    "LM10Potential":                  [],
    "NullPotential":                  [],

    "NFW": [
        {"key": "m",   "label": "Mass m (Msun)",             "default": 1e12, "min": 1e8,  "max": 1e14},
        {"key": "r_s", "label": "Scale radius r_s (kpc)",    "default": 15.0, "min": 0.1,  "max": 200.0},
    ],
    "Hernquist": [
        {"key": "m", "label": "Mass m (Msun)",        "default": 1e10, "min": 1e6, "max": 1e13},
        {"key": "c", "label": "Scale radius c (kpc)", "default": 1.0,  "min": 0.01, "max": 100.0},
    ],
    "Plummer": [
        {"key": "m", "label": "Mass m (Msun)",        "default": 1e10, "min": 1e6, "max": 1e13},
        {"key": "b", "label": "Scale radius b (kpc)", "default": 1.0,  "min": 0.01, "max": 100.0},
    ],
    "Isochrone": [
        {"key": "m", "label": "Mass m (Msun)",        "default": 1e10, "min": 1e6, "max": 1e13},
        {"key": "b", "label": "Scale radius b (kpc)", "default": 1.0,  "min": 0.01, "max": 100.0},
    ],
    "Jaffe": [
        {"key": "m", "label": "Mass m (Msun)",        "default": 1e10, "min": 1e6, "max": 1e13},
        {"key": "c", "label": "Scale radius c (kpc)", "default": 1.0,  "min": 0.01, "max": 100.0},
    ],
    "Burkert": [
        {"key": "rho", "label": "Central density ρ (Msun/kpc³)", "default": 1e7,  "min": 1e3, "max": 1e11},
        {"key": "r0",  "label": "Core radius r0 (kpc)",          "default": 10.0, "min": 0.1, "max": 200.0},
    ],
    "Kepler": [
        {"key": "m", "label": "Mass m (Msun)", "default": 1e10, "min": 1e6, "max": 1e13},
    ],

    "MiyamotoNagai": [
        {"key": "m", "label": "Mass m (Msun)",   "default": 1e10, "min": 1e6,  "max": 1e13},
        {"key": "a", "label": "Scale a (kpc)",   "default": 3.0,  "min": 0.1,  "max": 50.0},
        {"key": "b", "label": "Scale b (kpc)",   "default": 0.3,  "min": 0.01, "max": 10.0},
    ],
    "MN3ExponentialDisk": [
        {"key": "m",   "label": "Mass m (Msun)",          "default": 1e10, "min": 1e6,  "max": 1e13},
        {"key": "h_R", "label": "Radial scale h_R (kpc)", "default": 3.0,  "min": 0.1,  "max": 50.0},
        {"key": "h_z", "label": "Height scale h_z (kpc)", "default": 0.3,  "min": 0.01, "max": 10.0},
    ],
    "Kuzmin": [
        {"key": "m", "label": "Mass m (Msun)", "default": 1e10, "min": 1e6, "max": 1e13},
        {"key": "a", "label": "Scale a (kpc)", "default": 3.0,  "min": 0.1, "max": 50.0},
    ],

    "Logarithmic": [
        {"key": "v_c", "label": "Circular velocity v_c (kpc/Myr; 0.2 ≈ 196 km/s)", "default": 0.2,  "min": 0.001, "max": 2.0},
        {"key": "r_h", "label": "Core radius r_h (kpc)",                            "default": 10.0, "min": 0.1,   "max": 200.0},
    ],
    "LeeSutoTriaxialNFW": [
        {"key": "v_c", "label": "Circular velocity v_c (kpc/Myr; 0.2 ≈ 196 km/s)", "default": 0.2,  "min": 0.001, "max": 2.0},
        {"key": "r_s", "label": "Scale radius r_s (kpc)",                           "default": 20.0, "min": 0.1,   "max": 200.0},
        {"key": "b",   "label": "Axis ratio b/a",                                   "default": 0.9,  "min": 0.1,   "max": 1.0},
        {"key": "c",   "label": "Axis ratio c/a",                                   "default": 0.8,  "min": 0.1,   "max": 1.0},
    ],
    "LongMuraliBar": [
        {"key": "m",     "label": "Mass m (Msun)",             "default": 1e10, "min": 1e6,      "max": 1e13},
        {"key": "a",     "label": "Major axis a (kpc)",        "default": 10.0, "min": 0.1,      "max": 100.0},
        {"key": "b",     "label": "Intermediate axis b (kpc)", "default": 3.0,  "min": 0.1,      "max": 50.0},
        {"key": "c",     "label": "Minor axis c (kpc)",        "default": 1.0,  "min": 0.01,     "max": 20.0},
        {"key": "alpha", "label": "Rotation angle α (rad)",    "default": 0.0,  "min": -3.14159, "max": 3.14159},
    ],
}

# ──────────────────────────────────────────────
# Constructors: callable that takes **params and returns a potential instance.
# The **_ absorbs any extra keys (e.g. from old serialised jobs).
# ──────────────────────────────────────────────

POTENTIAL_CONSTRUCTORS = {
    "MilkyWayPotential v2 (default)":
        lambda **_: gp.MilkyWayPotential(version="v2"),
    "MilkyWayPotential v1":
        lambda **_: gp.MilkyWayPotential(version="v1"),
    "BovyMWPotential2014":
        lambda **_: gp.BovyMWPotential2014(),
    "LM10Potential":
        lambda **_: gp.LM10Potential(),
    "NullPotential":
        lambda **_: gp.NullPotential(),

    "NFW":
        lambda m=1e12, r_s=15, **_: gp.NFWPotential(m=m, r_s=r_s, units=galactic),
    "Hernquist":
        lambda m=1e10, c=1, **_: gp.HernquistPotential(m=m, c=c, units=galactic),
    "Plummer":
        lambda m=1e10, b=1, **_: gp.PlummerPotential(m=m, b=b, units=galactic),
    "Isochrone":
        lambda m=1e10, b=1, **_: gp.IsochronePotential(m=m, b=b, units=galactic),
    "Jaffe":
        lambda m=1e10, c=1, **_: gp.JaffePotential(m=m, c=c, units=galactic),
    "Burkert":
        lambda rho=1e7, r0=10, **_: gp.BurkertPotential(rho=rho, r0=r0, units=galactic),
    "Kepler":
        lambda m=1e10, **_: gp.KeplerPotential(m=m, units=galactic),

    "MiyamotoNagai":
        lambda m=1e10, a=3, b=0.3, **_: gp.MiyamotoNagaiPotential(m=m, a=a, b=b, units=galactic),
    "MN3ExponentialDisk":
        lambda m=1e10, h_R=3, h_z=0.3, **_: gp.MN3ExponentialDiskPotential(m=m, h_R=h_R, h_z=h_z, units=galactic),
    "Kuzmin":
        lambda m=1e10, a=3, **_: gp.KuzminPotential(m=m, a=a, units=galactic),

    "Logarithmic":
        lambda v_c=0.2, r_h=10, **_: gp.LogarithmicPotential(v_c=v_c, r_h=r_h, q1=1, q2=1, q3=1, units=galactic),
    "LeeSutoTriaxialNFW":
        lambda v_c=0.2, r_s=20, b=0.9, c=0.8, **_: gp.LeeSutoTriaxialNFWPotential(v_c=v_c, r_s=r_s, a=1, b=b, c=c, units=galactic),
    "LongMuraliBar":
        lambda m=1e10, a=10, b=3, c=1, alpha=0, **_: gp.LongMuraliBarPotential(m=m, a=a, b=b, c=c, alpha=alpha, units=galactic),
}

POTENTIAL_NAMES = list(POTENTIAL_CONSTRUCTORS.keys())

# ──────────────────────────────────────────────
# Code-generation templates  (used by make_potential_code)
# Placeholders match the keys in POTENTIAL_PARAMS.
# ──────────────────────────────────────────────

_POTENTIAL_CODE_TEMPLATES = {
    "MilkyWayPotential v2 (default)": "gp.MilkyWayPotential(version='v2')",
    "MilkyWayPotential v1":           "gp.MilkyWayPotential(version='v1')",
    "BovyMWPotential2014":            "gp.BovyMWPotential2014()",
    "LM10Potential":                  "gp.LM10Potential()",
    "NullPotential":                  "gp.NullPotential()",

    "NFW":
        "gp.NFWPotential(m={m:.6g}, r_s={r_s:.6g}, units=galactic)",
    "Hernquist":
        "gp.HernquistPotential(m={m:.6g}, c={c:.6g}, units=galactic)",
    "Plummer":
        "gp.PlummerPotential(m={m:.6g}, b={b:.6g}, units=galactic)",
    "Isochrone":
        "gp.IsochronePotential(m={m:.6g}, b={b:.6g}, units=galactic)",
    "Jaffe":
        "gp.JaffePotential(m={m:.6g}, c={c:.6g}, units=galactic)",
    "Burkert":
        "gp.BurkertPotential(rho={rho:.6g}, r0={r0:.6g}, units=galactic)",
    "Kepler":
        "gp.KeplerPotential(m={m:.6g}, units=galactic)",

    "MiyamotoNagai":
        "gp.MiyamotoNagaiPotential(m={m:.6g}, a={a:.6g}, b={b:.6g}, units=galactic)",
    "MN3ExponentialDisk":
        "gp.MN3ExponentialDiskPotential(m={m:.6g}, h_R={h_R:.6g}, h_z={h_z:.6g}, units=galactic)",
    "Kuzmin":
        "gp.KuzminPotential(m={m:.6g}, a={a:.6g}, units=galactic)",

    "Logarithmic":
        "gp.LogarithmicPotential(v_c={v_c:.6g}, r_h={r_h:.6g}, q1=1, q2=1, q3=1, units=galactic)",
    "LeeSutoTriaxialNFW":
        "gp.LeeSutoTriaxialNFWPotential(v_c={v_c:.6g}, r_s={r_s:.6g}, a=1, b={b:.6g}, c={c:.6g}, units=galactic)",
    "LongMuraliBar":
        "gp.LongMuraliBarPotential(m={m:.6g}, a={a:.6g}, b={b:.6g}, c={c:.6g}, alpha={alpha:.6g}, units=galactic)",
}


def make_potential_code(name: str, params: dict | None = None) -> str:
    """Return a gala Python constructor string suitable for pasting into user code."""
    if params is None:
        params = {}
    template = _POTENTIAL_CODE_TEMPLATES.get(name, "gp.MilkyWayPotential(version='v2')")
    if not params:
        return template
    # Fill defaults from spec, then override with user-supplied values
    defaults = {s["key"]: s["default"] for s in POTENTIAL_PARAMS.get(name, [])}
    merged = {**defaults, **params}
    try:
        return template.format(**merged)
    except (KeyError, ValueError):
        return template
