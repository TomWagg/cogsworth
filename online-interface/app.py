"""
    This file defines a web interface to the cogsworth code using Streamlit

    The main purpose of this file is to define the Streamlit app, including all the widgets and plotting code.  The actual work of running simulations is done in worker.py, which defines helper functions that can be called from this app.py file.  This separation allows us to keep the Streamlit-specific code separate from the core logic of setting up and running simulations, which also makes it easier to test the simulation code independently of the web interface.
"""

import ast
import json
import os
import re
import tempfile
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
import pandas as pd
from astropy import units as u
import plotly.express as px

import cosmic
import cogsworth.pop
from cogsworth.utils import get_default_BSE_settings, kstar_translator
import queue_db
from potentials import POTENTIAL_NAMES, POTENTIAL_PARAMS, make_potential_code

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

KSTAR_NAMES = {
    i: f"{i} - {kstar_translator[i]['long']}" for i in range(16)
}
KSTAR_OPTIONS = list(KSTAR_NAMES.values())
KSTAR_TO_INT  = {v: k for k, v in KSTAR_NAMES.items()}

SFH_CLASS_CODE = {
    "Wagg+2022 (default)":     "sfh.Wagg2022",
    "Burst Uniform Disc":      "sfh.BurstUniformDisc",
    "Constant Uniform Disc":   "sfh.ConstantUniformDisc",
    "Constant Plummer Sphere": "sfh.ConstantPlummerSphere",
    "Sanders & Binney 2015":   "sfh.SandersBinney2015",
    "Carina Dwarf":            "sfh.CarinaDwarf",
}
SFH_MODEL_NAMES = list(SFH_CLASS_CODE.keys())

BSE_DEFAULTS = get_default_BSE_settings()
BSE_OVERRIDABLE_KEYS = [k for k in BSE_DEFAULTS if k != "binfrac"]

WORKER_WARN_SECS = 30

_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}

# ──────────────────────────────────────────────
# Page config & theme
# ──────────────────────────────────────────────

# add some custom CSS
st.markdown(
    """
    <style>
    div[role="tablist"] button {
        width: 100%
    }
    div[role="tablist"] button p {
        font-size: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(
    page_title="cogsworth",
    page_icon=":material/wand_stars:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# add the cogsworth logo image
st.logo("../docs/_static/cogsworth-flat.png")

st.title("`cogsworth` online", text_alignment='center')
st.markdown("Welcome to the online interface for `cogsworth`! Dive straight in below, or use these dropdowns to learn more!", text_alignment='center')

col1, col2 = st.columns([1, 1])
with col1:
    with st.expander("What's `cogsworth`?"):
        _, smallcol, _ = st.columns([1, 2, 1])
        with smallcol:
            st.image("../docs/_static/cogsworth-logo.png", width='content')
        st.subheader("Self-consistent binary population synthesis & galactic dynamics simulations", text_alignment='center')
        st.markdown(
            "`cogsworth` is a Python package for simulating the evolution of binary star populations within a galaxy, with self-consistent stellar evolution and galactic dynamics. It combines `COSMIC` for binary population synthesis with `gala` for galactic dynamics, allowing you track the evolution of binaries in their galactic context, predicting present day kinematics in addition to intrinsic binary properties."
        )
        st.markdown(
            "You can learn more about cogsworth and its capabilites using the buttons below!"
        )

        doc_col1, doc_col2 = st.columns([1, 1])
        with doc_col1:
            st.link_button(
                "cogsworth documentation",
                "https://cogsworth.readthedocs.io/",
                icon=":material/book_2:",
                width='stretch'
            )

        with doc_col2:
            st.link_button(
                "GitHub Repo",
                "https://github.com/TomWagg/cogsworth",
                icon=":material/code:",
                width='stretch'
            )


with col2:
    with st.expander("What can I use this online interface for?"):
        st.markdown(
            "This interface allows you to run binary population synthesis simulations with self-consistent galactic dynamics, all from your web browser. Use the tabs above to set up a population or single binary simulation, and explore the results with interactive tables and plots. There's also an option for converting your simulation to a code block."
        )
        st.markdown(
            "Note that you should use the full Python package for research quality simulations! The goal of this interface is to let people check out the capabilities of the code, learn something about population synthesis, and generate quick results without any installation."
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


@st.cache_data
def _load_sampling_spec() -> list:
    """Load the COSMIC sampling settings spec from the bundled JSON, cached once."""
    from importlib.resources import files
    raw = json.loads((files("cosmic") / "data" / "cosmic-settings.json").read_text())
    for section in raw:
        if isinstance(section, dict) and section.get("category") == "sampling":
            return section["settings"]
    return []


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def sampling_params_form(key_prefix: str = "samp") -> dict:
    """Render the COSMIC initial-sampling parameters expander.

    Loads the parameter spec dynamically from the COSMIC JSON so any update
    to COSMIC is reflected automatically.  Returns a dict of non-default values
    suitable for passing directly to ``Population(sampling_params=...)``.
    Always uses independent sampling; sampling_method and metallicity are hidden.
    """
    specs = _load_sampling_spec()

    # Always use independent sampling; hide these from the UI entirely
    HIDE = {"sampling_method", "metallicity"}

    def _default(spec):
        for opt in spec.get("options", []):
            if opt.get("default"):
                return opt["name"]
        opts = spec.get("options", [])
        return opts[0]["name"] if opts else None

    params: dict = {}

    with st.expander("Initial sampling parameters", expanded=False):
        st.caption(
            "Change how the initial binary population is sampled. "
            "See the [COSMIC documentation](https://cosmic-popsynth.github.io/COSMIC/pages/sampling.html) "
            "for details on each parameter."
        )

        for spec in specs:
            name = spec["name"]
            if name in HIDE:
                continue

            help_text = _strip_html(spec.get("description", ""))
            stype     = spec.get("type", "number")
            default   = _default(spec)

            if name == "binfrac_model":
                # Named model or a custom numeric fraction via slider
                named_opts = ["vanHaaften", "offner23"]
                choices    = named_opts + ["Custom fraction"]
                choice = st.selectbox(
                    "binfrac_model", choices,
                    index=len(choices) - 1,  # default: Custom fraction (0.5)
                    key=f"{key_prefix}_binfrac_choice",
                    help=help_text,
                )
                if choice == "Custom fraction":
                    frac = st.slider(
                        "Binary fraction", 0.0, 1.0, 1.0, 0.01,
                        key=f"{key_prefix}_binfrac_slider",
                    )
                    params["binfrac_model"] = frac
                else:
                    params["binfrac_model"] = choice

            elif stype == "dropdown":
                opts = [o["name"] for o in spec["options"]]
                default_idx = next(
                    (i for i, o in enumerate(spec["options"]) if o.get("default")), 0
                )
                val = st.selectbox(
                    name, opts, index=default_idx,
                    key=f"{key_prefix}_{name}", help=help_text,
                )
                if val != opts[default_idx]:
                    params[name] = val

            elif stype == "number":
                if default is None or default == "None":
                    raw = st.text_input(
                        name, value="",
                        placeholder="None (default)",
                        key=f"{key_prefix}_{name}", help=help_text,
                    )
                    if raw.strip():
                        try:
                            params[name] = float(raw)
                        except ValueError:
                            st.warning(f"Invalid numeric value for {name}: {raw!r}")
                else:
                    val = st.number_input(
                        name, value=float(default), format="%.6g",
                        key=f"{key_prefix}_{name}", help=help_text,
                    )
                    if val != float(default):
                        params[name] = val

    return params


def sfh_params_form(model_name: str) -> dict:
    """Render SFH-specific widgets; return params dict with astropy Quantities."""
    params = {}
    if model_name == "Wagg+2022 (default)":
        with st.expander(f"Star-formation history parameters ({model_name})", expanded=False):
            params["galaxy_age"] = st.slider("Galaxy age (Gyr)", 8.0, 14.0, 12.0, 0.5, key="w22_age") * u.Gyr
            params["tsfr"] = st.slider("Star-formation timescale tsfr (Gyr)", 1.0, 12.0, 6.8, 0.1, key="w22_tsfr") * u.Gyr

    elif model_name in ("Burst Uniform Disc", "Constant Uniform Disc"):
        p = "bud" if "Burst" in model_name else "cud"
        with st.expander(f"Star-formation history parameters ({model_name})", expanded=False):
            params["t_burst"] = st.slider("Birth time (Gyr ago)", 0.0, 12.0, 6.0, 0.5, key=f"{p}_tburst") * u.Gyr
            params["z_max"]   = st.number_input("Max height z_max (kpc)", 0.1, 5.0, 2.0, key=f"{p}_zmax") * u.kpc
            params["R_max"]   = st.number_input("Max disc radius R_max (kpc)", 1.0, 30.0, 15.0, key=f"{p}_Rmax") * u.kpc
            if st.checkbox("Fix metallicity", key=f"{p}_fixZ"):
                params["Z_all"] = st.number_input("Metallicity", 1e-4, 0.03, 0.02, format="%.4f", key=f"{p}_Z")

    elif model_name == "Constant Plummer Sphere":
        with st.expander(f"Star-formation history parameters ({model_name})", expanded=False):
            params["tau_min"] = st.slider("Min lookback time (Gyr)", 0.0, 12.0, 0.0, 0.5, key="cps_tmin") * u.Gyr
            params["tau_max"] = st.slider("Max lookback time (Gyr)", 0.0, 12.0, 12.0, 0.5, key="cps_tmax") * u.Gyr
            params["Z_all"]   = st.number_input("Metallicity", 1e-4, 0.03, 0.02, format="%.4f", key="cps_Z")
            params["M"]       = st.number_input("Total mass (x 1e10 Msun)", 0.1, 100.0, 1.0, key="cps_M") * 1e10 * u.Msun
            params["a"]       = st.number_input("Scale radius a (kpc)", 0.1, 10.0, 1.0, key="cps_a") * u.kpc

    elif model_name == "Sanders & Binney 2015":
        with st.expander(f"Star-formation history parameters ({model_name})", expanded=False):
            params["time_bins"] = st.slider("Time bins", 2, 10, 5, key="sb_bins")

    elif model_name == "Carina Dwarf":
        st.info("Using default Carina Dwarf parameters.")

    return params


def potential_params_form(pot_name: str, key_prefix: str = "pot") -> dict:
    """Render potential-specific parameter widgets; return dict of param values."""
    specs = POTENTIAL_PARAMS.get(pot_name, [])
    if not specs:
        return {}
    with st.expander(f"Galactic potential parameters ({pot_name})", expanded=False):
        params = {}
        for spec in specs:
            key = spec["key"]
            minv, maxv, default = spec["min"], spec["max"], spec["default"]
            # Use a reasonable step: 1/100 of the range, clamped to sensible precision
            raw_step = (maxv - minv) / 100.0
            # Round step to 1 sig fig
            import math
            if raw_step > 0:
                mag = 10 ** math.floor(math.log10(raw_step))
                step = round(raw_step / mag) * mag
            else:
                step = 0.01
            val = st.number_input(
                spec["label"],
                min_value=float(minv),
                max_value=float(maxv),
                value=float(default),
                step=float(step),
                format="%.6g",
                key=f"{key_prefix}_pot_{key}",
            )
            params[key] = float(val)
    return params


def bse_settings_form(key_prefix: str = "bse") -> dict:
    """Render an add/remove interface for BSE setting overrides.

    Returns the current overrides dict (stored in session_state).
    """
    overrides: dict = st.session_state.setdefault("bse_overrides", {})

    with st.expander("Binary physics settings", expanded=False):
        st.caption(
            "Alter any default COSMIC BSE parameter. "
            "Array-valued parameters (e.g. qcrit_array) accept JSON lists.\n\n"
            "To learn about what each parameter does, check out the [COSMIC documentation](https://cosmic-popsynth.github.io/COSMIC/pages/inifile.html)."
        )

        # ── Display current overrides ──────────────────────────────────────
        if overrides:
            st.markdown("**Altered settings:**")
            for k in list(overrides):
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.text(k)
                with c2:
                    st.text(repr(overrides[k]))
                with c3:
                    if st.button("Remove", key=f"{key_prefix}_rm_{k}"):
                        del overrides[k]
                        st.rerun()
        else:
            st.markdown("*Currently no updated settings — all BSE settings use their defaults.*", text_alignment='center')

        st.divider()

        # ── Add new override ───────────────────────────────────────────────
        st.markdown("**Change a binary physics setting:**")
        add_col1, add_col2, add_col3 = st.columns([3, 2, 1])
        with add_col1:
            sel_key = st.selectbox(
                "Parameter",
                ["(choose)"] + BSE_OVERRIDABLE_KEYS,
                key=f"{key_prefix}_sel",
            )
        if sel_key != "(choose)":
            default_val = BSE_DEFAULTS[sel_key]
            with add_col2:
                if isinstance(default_val, list):
                    raw = st.text_input(
                        f"Value (JSON list, default: ...)",
                        value=json.dumps(default_val),
                        key=f"{key_prefix}_v_{sel_key}",
                    )
                else:
                    raw = st.number_input(
                        f"Value (default: {default_val})",
                        value=float(default_val),
                        format="%.6g",
                        key=f"{key_prefix}_v_{sel_key}",
                    )
            with add_col3:
                st.write("")  # vertical align
                if st.button("Add", key=f"{key_prefix}_add"):
                    if isinstance(default_val, list):
                        try:
                            overrides[sel_key] = json.loads(raw)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON list.")
                    else:
                        overrides[sel_key] = type(default_val)(raw)
                    st.rerun()

        st.divider()

        # ── Bulk import ────────────────────────────────────────────────────
        st.markdown("**Import settings from file or dict:**")
        imp_tab1, imp_tab2 = st.tabs(["Upload Params.ini", "Paste BSE_settings dict"])

        with imp_tab1:
            uploaded = st.file_uploader(
                "Params.ini", type=["ini"], label_visibility="collapsed",
                key=f"{key_prefix}_ini_upload",
            )
            if uploaded is not None:
                if st.button("Apply", key=f"{key_prefix}_ini_apply"):
                    try:
                        from cosmic.utils import parse_inifile
                        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False, mode="wb") as f:
                            f.write(uploaded.read())
                            tmp_path = f.name
                        bse_dict = parse_inifile(tmp_path)[0]
                        os.unlink(tmp_path)
                        count = 0
                        for k, v in bse_dict.items():
                            if k not in BSE_DEFAULTS:
                                continue
                            if v != BSE_DEFAULTS[k]:
                                overrides[k] = v
                                count += 1
                        st.success(f"Applied {count} non-default setting(s) from Params.ini.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to parse Params.ini: {e}")

        with imp_tab2:
            pasted = st.text_area(
                "BSE_settings dict", placeholder='{"alpha1": 2.0, "remnantflag": 5, ...}',
                key=f"{key_prefix}_dict_paste", label_visibility="collapsed",
            )
            if st.button("Apply", key=f"{key_prefix}_dict_apply"):
                if pasted.strip():
                    try:
                        d = ast.literal_eval(pasted.strip())
                        if not isinstance(d, dict):
                            raise ValueError("Expected a dict literal.")
                        count = 0
                        for k, v in d.items():
                            if k not in BSE_DEFAULTS:
                                st.error(f"Unknown BSE setting: {k}, skipping")
                                continue
                            # check if value is different from default (after converting types)
                            if ((isinstance(BSE_DEFAULTS[k], list) and not np.allclose(v, BSE_DEFAULTS[k])) or
                                (v != BSE_DEFAULTS[k])):
                                overrides[k] = v
                                count += 1
                        print("Got here", count)
                        st.success(f"Applied {count} setting(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not parse dict: {e}")

    return overrides


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def clear_job():
    for k in ("job_id", "pop", "pop_has_orbits", "last_job_type",
              "last_job_params", "max_ev_time", "pop_h5_bytes", "bi_filters",
              "pop_plots", "de_next_id", "bi_plots", "bi_next_id"):
        st.session_state.pop(k, None)


def _get_ig_df(pop) -> pd.DataFrame:
    """Build a filterable DataFrame of birth positions/velocities from _initial_galaxy."""
    ig  = pop._initial_galaxy
    idx = pop.bin_nums
    x   = ig._x.to(u.kpc).value
    y   = ig._y.to(u.kpc).value
    z   = ig._z.to(u.kpc).value
    tau = ig._tau.to(u.Gyr).value
    vR  = ig.v_R.to(u.km / u.s).value
    vT  = ig.v_T.to(u.km / u.s).value
    vz  = ig.v_z.to(u.km / u.s).value
    Z   = ig._Z
    # Sun at (8.122, 0, 0.027) kpc galactocentric
    dist_sun = np.sqrt((x - 8.122)**2 + y**2 + (z - 0.027)**2)
    dist_gc  = np.sqrt(x**2 + y**2 + z**2)
    return pd.DataFrame({
        "x_kpc": x, "y_kpc": y, "z_kpc": z,
        "tau_gyr": tau, "Z": Z,
        "dist_sun_kpc": dist_sun,
        "dist_gc_kpc":  dist_gc,
        "v_R_kms": vR, "v_T_kms": vT, "v_z_kms": vz,
    }, index=idx)


def apply_filters(pop, filters: list) -> set:
    """Return the set of bin_nums satisfying ALL conditions in filters."""
    matching = set(pop.bin_nums)
    for f in filters:
        source, col, op, val_str = f["source"], f["col"], f["op"], f["val"]
        op_fn = _OPS.get(op)
        if op_fn is None:
            continue
        try:
            val = float(val_str)
        except ValueError:
            val = val_str
        if source == "final_bpp":
            df   = pop.final_bpp
            mask = op_fn(df[col], val)
            match = set(df.index[mask])
        elif source == "initial_binaries":
            df   = pop.initial_binaries
            mask = op_fn(df[col], val)
            match = set(df.index[mask])
        elif source == "bpp (any row)":
            bpp = pop.bpp
            grouped = (bpp.groupby(level=0)
                        if isinstance(bpp.index, pd.MultiIndex)
                        else bpp.groupby(bpp.index))
            match = {bn for bn, grp in grouped if op_fn(grp[col], val).any()}
        elif source == "initial_galaxy":
            df = _get_ig_df(pop)
            mask = op_fn(df[col], val)
            match = set(df.index[mask])
        else:
            match = set(pop.bin_nums)
        matching &= match
    return matching


def _plot_theme():
    """Return Plotly colour settings matching the current Streamlit theme."""
    is_dark = True  # safe default
    # st.context.theme.backgroundColor reflects the actual active theme
    # (including user overrides from the hamburger menu).  Using brightness
    # rather than .base avoids the problem where config.toml sets dark
    # colours without setting base = "dark", which makes .base return
    # "light" even for a visually dark theme.
    is_dark = st.context.theme.type == "dark"

    if is_dark:
        return dict(
            is_dark=True,
            plot_bgcolor="#1a192d",
            font_color="#e5e2f5",
            grid_color="#2a2940",
            zeroline_color="#3a3860",
            pri_color="#045993",
            sec_color="#db6100",
            start_color="white",
            start_border="#666",
            merger_color="white",
        )
    else:
        return dict(
            is_dark=False,
            plot_bgcolor="#f5f5ff",
            font_color="#111111",
            grid_color="#ddddee",
            zeroline_color="#aaaacc",
            pri_color="#045993",
            sec_color="#db6100",
            start_color="#333333",
            start_border="#aaa",
            merger_color="#333333",
        )


def _plot_orbit_plotly(pop, bin_num, t_min_yr, t_max_yr, theme, animated=False):
    """
    Build a Plotly orbit figure.
    animated=False → static: shows the full path in the time range at once.
    animated=True  → starts empty, grows the path with Play/Pause controls.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ind = int(np.where(pop.bin_nums == bin_num)[0][0])
    disrupted = bool(pop.disrupted[ind])
    primary_orbit = pop.primary_orbits[ind]
    secondary_orbit = pop.secondary_orbits[ind] if disrupted else None

    # time since birth in years (orbit.t is in Myr in gala galactic units)
    t0 = primary_orbit.t[0]
    t_since_birth_yr = (primary_orbit.t - t0).to(u.yr).value

    mask = (t_since_birth_yr >= t_min_yr) & (t_since_birth_yr <= t_max_yr)
    if not mask.any():
        mask[0] = True

    px_kpc = primary_orbit.x.to(u.kpc).value[mask]
    py_kpc = primary_orbit.y.to(u.kpc).value[mask]
    pz_kpc = primary_orbit.z.to(u.kpc).value[mask]
    t_plot_yr = t_since_birth_yr[mask]

    sx_kpc = sy_kpc = sz_kpc = None
    if secondary_orbit is not None:
        sx_kpc = secondary_orbit.x.to(u.kpc).value[mask]
        sy_kpc = secondary_orbit.y.to(u.kpc).value[mask]
        sz_kpc = secondary_orbit.z.to(u.kpc).value[mask]

    # ── subplots ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=3,
        horizontal_spacing=0.10,
    )

    pri_color   = theme["pri_color"]
    sec_color   = theme["sec_color"]
    start_color = theme["start_color"]

    t_plot_myr = t_plot_yr / 1e6

    # For the animation, orbit traces start empty and fill via frames.
    # For the static plot, they show the full path immediately.
    init_px, init_py, init_pz = ([], [], []) if animated else (px_kpc, py_kpc, pz_kpc)
    init_t = [] if animated else t_plot_myr
    init_sx = init_sy = init_sz = []
    if secondary_orbit is not None and not animated:
        init_sx, init_sy, init_sz = sx_kpc, sy_kpc, sz_kpc

    def _add_line(xs, ys, zs, name, color, col, showlegend, xlabel, ylabel, zlabel):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name, showlegend=showlegend,
            legendgroup=name,
            line=dict(color=color, width=1.5),
            customdata=np.column_stack([zs, init_t]) if len(xs) else np.empty((0, 2)),
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"{xlabel}: %{{x:.3f}} kpc<br>"
                f"{ylabel}: %{{y:.3f}} kpc<br>"
                f"{zlabel}: %{{customdata[0]:.3f}} kpc<br>"
                "t = %{customdata[1]:.0f} Myr"
                "<extra></extra>"
            ),
        ), row=1, col=col)

    # panels: (col, x-array, y-array, z-array, x-label, y-label, z-label)
    panels = [
        (1, "x", "y", "z"),
        (2, "x", "z", "y"),
        (3, "y", "z", "x"),
    ]
    coord = {"x": (px_kpc, sx_kpc), "y": (py_kpc, sy_kpc), "z": (pz_kpc, sz_kpc)}
    init_coord = {"x": init_px, "y": init_py, "z": init_pz}
    init_s = {"x": init_sx, "y": init_sy, "z": init_sz}

    for col, xl, yl, zl in panels:
        first = col == 1
        _add_line(init_coord[xl], init_coord[yl], init_coord[zl],
                  "Primary", pri_color, col, first, xl, yl, zl)
        if secondary_orbit is not None:
            _add_line(init_s[xl], init_s[yl], init_s[zl],
                      "Secondary", sec_color, col, first, xl, yl, zl)

    n_orbit_traces = 3 + (3 if secondary_orbit is not None else 0)

    def _add_point(x, y, z, t_myr, name, marker_kw, col, showlegend, xl, yl, zl):
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name=name,
            showlegend=showlegend,
            legendgroup=name,
            marker=marker_kw,
            customdata=[[z, t_myr]],
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"{xl}: %{{x:.3f}} kpc<br>"
                f"{yl}: %{{y:.3f}} kpc<br>"
                f"{zl}: %{{customdata[0]:.3f}} kpc<br>"
                "t = %{customdata[1]:.0f} Myr"
                "<extra></extra>"
            ),
        ), row=1, col=col)

    # birth-position markers
    birth_marker = dict(color=start_color, size=9, symbol="circle",
                        line=dict(color=theme["start_border"], width=1))
    for col, xl, yl, zl in panels:
        xv, yv, zv = coord[xl][0][0], coord[yl][0][0], coord[zl][0][0]
        _add_point(xv, yv, zv, 0.0, "Birth position", birth_marker,
                   col, col == 1, xl, yl, zl)

    # ── SN / merger markers (static) ──────────────────────────────────────
    rows_bpp = pop.bpp.loc[bin_num]
    events = [
        (
            (rows_bpp["evol_type"] == 15)
            | ((rows_bpp["evol_type"] == 16) & (rows_bpp["sep"] == 0.0)),
            primary_orbit, "Primary SN", "star", theme["pri_color"],
        ),
        (
            (rows_bpp["evol_type"] == 16) & (rows_bpp["sep"] != 0.0),
            secondary_orbit if disrupted else None,
            "Secondary SN", "star", theme["sec_color"],
        ),
        (rows_bpp["sep"] == 0.0, primary_orbit, "Merger", "x", theme["merger_color"]),
    ]
    for ev_mask, ev_orbit, ev_name, ev_symbol, ev_color in events:
        if not np.any(ev_mask) or ev_orbit is None:
            continue
        ev_time_myr = float(rows_bpp["tphys"][ev_mask].iloc[0])
        if ev_time_myr * 1e6 < t_min_yr or ev_time_myr * 1e6 > t_max_yr:
            continue
        ev_t_myr = (ev_orbit.t - ev_orbit.t[0]).to(u.Myr).value
        ev_idx   = np.searchsorted(ev_t_myr, ev_time_myr, side="right") - 1
        ecoord   = {
            "x": float(ev_orbit.x[ev_idx].to(u.kpc).value),
            "y": float(ev_orbit.y[ev_idx].to(u.kpc).value),
            "z": float(ev_orbit.z[ev_idx].to(u.kpc).value),
        }
        ev_marker = dict(color=ev_color, size=12, symbol=ev_symbol)
        for col, xl, yl, zl in panels:
            _add_point(ecoord[xl], ecoord[yl], ecoord[zl], ev_time_myr,
                       ev_name, ev_marker, col, col == 1, xl, yl, zl)

    # ── animation frames (only when animated=True) ────────────────────────
    extra_layout = {}
    if animated:
        n_steps   = len(px_kpc)
        n_frames  = min(n_steps, 120)
        frame_idx = np.linspace(0, n_steps - 1, n_frames, dtype=int)

        def _paired_ranges(a_vals, b_vals, pad=0.12):
            """Equal-span [lo, hi] ranges for two axes, each independently centered."""
            a_c = (a_vals.min() + a_vals.max()) / 2
            b_c = (b_vals.min() + b_vals.max()) / 2
            half = max(
                (a_vals.max() - a_vals.min()) / 2,
                (b_vals.max() - b_vals.min()) / 2,
                0.1,
            ) * (1 + pad)
            return [a_c - half, a_c + half], [b_c - half, b_c + half]

        frames = []
        for fi in frame_idx:
            sl  = slice(fi + 1)
            t_s = t_plot_myr[:fi+1]
            pdata = {"x": px_kpc[sl], "y": py_kpc[sl], "z": pz_kpc[sl]}
            sdata = {"x": sx_kpc[sl], "y": sy_kpc[sl], "z": sz_kpc[sl]} if secondary_orbit is not None else None

            # Build frame data in the same interleaved order traces were added
            fd = []
            for _, xl, yl, zl in panels:
                fd.append(go.Scatter(
                    x=pdata[xl], y=pdata[yl],
                    customdata=np.column_stack([pdata[zl], t_s]),
                ))
                if secondary_orbit is not None:
                    fd.append(go.Scatter(
                        x=sdata[xl], y=sdata[yl],
                        customdata=np.column_stack([sdata[zl], t_s]),
                    ))

            frames.append(go.Frame(
                data=fd,
                traces=list(range(n_orbit_traces)),
                name=str(fi),
            ))
        fig.frames = frames

        slider_steps = [
            dict(
                method="animate",
                args=[[f.name], {"mode": "immediate",
                                 "frame": {"duration": 30, "redraw": True},
                                 "transition": {"duration": 0}}],
                label=f"{t_plot_yr[int(f.name)]:.2e}",
            )
            for f in frames
        ]
        extra_layout = dict(
            margin=dict(t=60, b=120),
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.5, xanchor="center", y=-0.20, yanchor="top",
                font=dict(color=theme["font_color"]),
                bgcolor="#2a2940" if theme["is_dark"] else "#e8e8f8",
                buttons=[
                    dict(label="▶  Play", method="animate",
                         args=[None, {"frame": {"duration": 30, "redraw": True},
                                      "fromcurrent": True,
                                      "transition": {"duration": 0}}]),
                    dict(label="⏸  Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}]),
                ],
            )],
            sliders=[dict(
                steps=slider_steps,
                transition=dict(duration=0),
                x=0.0, y=-0.10, len=1.0,
                bgcolor="#2a2940" if theme["is_dark"] else "#e0e0f0",
                font=dict(color=theme["font_color"]),
                currentvalue=dict(prefix="t = ", suffix=" yr", visible=True,
                                  xanchor="center",
                                  font=dict(color=theme["font_color"])),
                pad=dict(b=10),
            )],
        )

    # ── shared layout ─────────────────────────────────────────────────────
    fig.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=theme["plot_bgcolor"],
        font=dict(color=theme["font_color"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.06,
                    xanchor="center", x=0.5),
        margin=extra_layout.pop("margin", dict(t=60, b=40)),
        **extra_layout,
    )

    axis_style = dict(
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["zeroline_color"],
    )
    # axis labels and equal aspect ratio per panel
    labels = [("x (kpc)", "y (kpc)"), ("x (kpc)", "z (kpc)"), ("y (kpc)", "z (kpc)")]
    x_axis_refs = ["x", "x2", "x3"]
    for col, ((xlabel, ylabel), xref) in enumerate(zip(labels, x_axis_refs), start=1):
        fig.update_xaxes(title_text=xlabel, constrain="domain",
                         row=1, col=col, **axis_style)
        fig.update_yaxes(title_text=ylabel, scaleanchor=xref, scaleratio=1,
                         constrain="domain", row=1, col=col, **axis_style)

    # For animation, fix axis limits to cover the full orbit so they never change
    if animated:
        x_all = px_kpc if secondary_orbit is None else np.concatenate([px_kpc, sx_kpc])
        y_all = py_kpc if secondary_orbit is None else np.concatenate([py_kpc, sy_kpc])
        z_all = pz_kpc if secondary_orbit is None else np.concatenate([pz_kpc, sz_kpc])
        xr_xy, yr_xy = _paired_ranges(x_all, y_all)
        xr_xz, yr_xz = _paired_ranges(x_all, z_all)
        xr_yz, yr_yz = _paired_ranges(y_all, z_all)
        fig.update_xaxes(range=xr_xy, row=1, col=1)
        fig.update_yaxes(range=yr_xy, row=1, col=1)
        fig.update_xaxes(range=xr_xz, row=1, col=2)
        fig.update_yaxes(range=yr_xz, row=1, col=2)
        fig.update_xaxes(range=xr_yz, row=1, col=3)
        fig.update_yaxes(range=yr_yz, row=1, col=3)

    return fig


def _rgba(tup):
    """Convert a (r, g, b, a) float tuple from kstar_translator to a CSS rgba string."""
    r, g, b, a = tup
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})"


def _downsample_bcm(df, max_points=2000):
    """
    Downsample a single-binary BCM DataFrame to at most max_points rows.
    Always keeps rows where kstar_1 or kstar_2 changes (evolutionary transitions),
    the first and last row; fills the remaining budget with a uniform stride
    over the non-transition rows.
    Uses numpy positional indices throughout to avoid pandas index/CoW issues.
    """
    n = len(df)
    if n <= max_points:
        return df

    # Build boolean keep mask in numpy — avoids any pandas copy-on-write issues
    keep = np.zeros(n, dtype=bool)
    keep[0]  = True
    keep[-1] = True

    for col in ("kstar_1", "kstar_2"):
        if col in df.columns:
            vals = df[col].to_numpy()
            keep[1:] |= vals[1:] != vals[:-1]

    n_keep  = keep.sum()
    n_extra = max(0, max_points - n_keep)

    if n_extra > 0:
        non_keep_pos = np.where(~keep)[0]
        stride       = max(1, len(non_keep_pos) // n_extra)
        keep[non_keep_pos[::stride]] = True

    return df.iloc[np.where(keep)[0]]


def _plot_hrd_plotly(pop, bin_num, theme, max_kstar=10):
    """Build an interactive Plotly HR diagram for a single binary."""
    import plotly.graph_objects as go

    bcm_bn = _downsample_bcm(pop.bcm.loc[bin_num])

    # We'll build one Scatter trace per (star, kstar) combination so each
    # stellar type gets a distinct colour and a single legend entry.
    fig = go.Figure()

    marker_color = theme["font_color"]

    for star_idx, (star_label, teff_col, lum_col, kstar_col) in enumerate([
        (1, "teff_1", "lum_1", "kstar_1"),
        (2, "teff_2", "lum_2", "kstar_2"),
    ]):
        teff  = bcm_bn[teff_col].values.astype(float)
        lum   = bcm_bn[lum_col].values.astype(float)
        kstar = bcm_bn[kstar_col].values.astype(int)

        mask = (kstar < max_kstar) & (teff > 0) & (lum > 0)
        if not mask.any():
            continue

        log_teff = np.log10(teff[mask])
        log_lum  = np.log10(lum[mask])
        kstar_m  = kstar[mask]

        # Faint connecting line
        fig.add_trace(go.Scatter(
            x=log_teff, y=log_lum,
            mode="lines",
            line=dict(color="grey", width=0.75),
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip",
            name=f"Star {star_idx + 1} track",
        ))

        # Coloured scatter points per kstar type
        seen_kstars = set()
        for k in np.unique(kstar_m):
            k_mask = kstar_m == k
            info   = kstar_translator[k]
            color  = _rgba(info["colour"])
            label  = info["short"]
            first  = label not in seen_kstars
            seen_kstars.add(label)
            fig.add_trace(go.Scatter(
                x=log_teff[k_mask], y=log_lum[k_mask],
                mode="markers",
                marker=dict(color=color, size=5),
                name=label,
                legendgroup=label,
                showlegend=first,
                hovertemplate=(
                    f"<b>{info['long']}</b><br>"
                    "log T<sub>eff</sub> = %{x:.3f}<br>"
                    "log L = %{y:.3f}<extra></extra>"
                ),
            ))

        # Start and end markers
        end_color   = marker_color if star_idx == 0 else "grey"
        end_opacity = 1.0           if star_idx == 0 else 0.7
        star_name   = "Primary" if star_idx == 0 else "Secondary"
        fig.add_trace(go.Scatter(
            x=[log_teff[0], log_teff[-1]],
            y=[log_lum[0],  log_lum[-1]],
            mode="markers",
            marker=dict(color=end_color, size=10, symbol="circle",
                        opacity=end_opacity,
                        line=dict(color=theme["grid_color"], width=1)),
            name=f"{star_name} start/end",
            hovertemplate=(
                "%{text}<br>"
                "log T<sub>eff</sub> = %{x:.3f}<br>"
                "log L = %{y:.3f}<extra></extra>"
            ),
            text=["Start", "End"],
        ))

    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=theme["plot_bgcolor"],
        font=dict(color=theme["font_color"]),
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top"),
        xaxis=dict(
            title="log10(Teff / K)",
            autorange="reversed",  # hotter stars on the left
            gridcolor=theme["grid_color"],
            zerolinecolor=theme["zeroline_color"],
        ),
        yaxis=dict(
            title="log10(L / Lsun)",
            gridcolor=theme["grid_color"],
            zerolinecolor=theme["zeroline_color"],
        ),
        margin=dict(t=40, r=160, b=60, l=60),
    )

    return fig


def _render_binary_inspector(pop, has_orbits: bool):
    """Render the Binary Inspector panel (filters + plot + info sidebar)."""

    if len(pop) > 1:
        sel_col, info_col = st.columns([3, 1])
    else:
        sel_col, info_col = None, st.container()  

    if len(pop) > 1:
        with sel_col:

            # ── Filter section ────────────────────────────────────────────────────
            with st.expander("Filter binaries", expanded=False):
                st.markdown("Use filters to narrow down the list of binaries.  You can filter on any column from the final_bpp, initial_binaries, or bpp tables, or on derived birth properties like birth position or metallicity.")
                filters: list = st.session_state.setdefault("bi_filters", [])

                if filters:
                    st.markdown("**Active filters:**")
                    for i, f in enumerate(list(filters)):
                        c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 2, 1])
                        with c1: st.text(f["source"])
                        with c2: st.text(f["col"])
                        with c3: st.text(f["op"])
                        with c4: st.text(str(f["val"]))
                        with c5:
                            if st.button("", key=f"bi_rm_filt_{i}", icon=':material/delete:'):
                                filters.pop(i); st.rerun()
                    if st.button("Clear all filters", key="bi_clr_all"):
                        filters.clear(); st.rerun()
                else:
                    st.caption("No filters active — showing all binaries.")

                st.divider()
                st.markdown("**Add a filter:**")

                IG_COLS = ["x_kpc", "y_kpc", "z_kpc", "tau_gyr", "Z",
                        "dist_sun_kpc", "dist_gc_kpc", "v_R_kms", "v_T_kms", "v_z_kms"]
                SOURCE_LABELS = {
                    "final_bpp":        "final_bpp  (final state)",
                    "initial_binaries": "initial_binaries  (birth parameters)",
                    "bpp (any row)":    "bpp  (any row — 'did it ever…')",
                    "initial_galaxy":   "initial_galaxy  (birth position/velocity)",
                }
                ac1, ac2, ac3, ac4, ac5 = st.columns([2, 2, 1, 2, 1])
                with ac1:
                    new_src_label = st.selectbox(
                        "Table", list(SOURCE_LABELS.values()), key="bi_f_src"
                    )
                    new_src = {v: k for k, v in SOURCE_LABELS.items()}[new_src_label]
                with ac2:
                    if new_src == "final_bpp":
                        col_opts = pop.final_bpp.columns.tolist()
                    elif new_src == "initial_binaries":
                        col_opts = pop.initial_binaries.columns.tolist()
                    elif new_src == "bpp (any row)":
                        col_opts = [c for c in pop.bpp.columns if c != "bin_num"]
                    else:
                        col_opts = IG_COLS
                    new_col = st.selectbox("Column", col_opts, key="bi_f_col")
                with ac3:
                    new_op = st.selectbox("Op", list(_OPS.keys()), key="bi_f_op")
                with ac4:
                    new_val = st.text_input("Value", key="bi_f_val", placeholder="e.g. 14")
                with ac5:
                    st.write("")
                    if st.button("Add", key="bi_f_add"):
                        if new_val.strip():
                            filters.append({"source": new_src, "col": new_col,
                                            "op": new_op, "val": new_val.strip()})
                            st.rerun()

            # ── Apply filters ─────────────────────────────────────────────────────
            all_bin_nums = list(pop.bin_nums)
            filters = st.session_state.get("bi_filters", [])
            display_bin_nums = all_bin_nums
            if filters:
                try:
                    matched = apply_filters(pop, filters)
                    display_bin_nums = [bn for bn in all_bin_nums if bn in matched]
                    n_match = len(display_bin_nums)
                    if n_match == 0:
                        st.warning("No binaries match the current filters — showing all.")
                        display_bin_nums = all_bin_nums
                    else:
                        st.info(f"**{n_match}** of {len(all_bin_nums)} binaries match the current filters.")
                except Exception as e:
                    st.warning(f"Filter error: {e}")

            if not display_bin_nums:
                st.warning("No binaries available.")
                return

            # ── Binary selector + final-state info ───────────────────────────────    
            selected_bn = st.selectbox("Select the binary that you want to inspect", display_bin_nums, key="bi_bn")
    else:
        selected_bn = pop.bin_nums[0]
    
    with info_col:
        row = pop.final_bpp.loc[[selected_bn]]
        with st.expander("Final state"):
            if len(pop) > 1:
                st.dataframe(row.T.rename(columns={row.index[0]: "value"}), width='stretch')
            else:
                print("ello")
                st.dataframe(row)

    # ── Add-plot toolbar ──────────────────────────────────────────────────
    st.markdown("#### Visualise your binary!")
    VIZ_TYPES = [
        "Cartoon binary evolution",
        "Detailed binary evolution",
        "Hertzsprung-Russell Diagram",
        "Galactic orbit",
    ]
    bi_plots = st.session_state.setdefault("bi_plots", [])
    ba, bb = st.columns([3, 6])
    with ba:
        add_viz = st.selectbox("", VIZ_TYPES, key="bi_add_type", label_visibility="collapsed")
    with bb:
        bc, bd, be = st.columns([1, 1, 2.4])
        with bc:
            if st.button("Add plot (full-width)", key="bi_add_btn", icon=":material/add:"):
                pid = st.session_state.get("bi_next_id", 0)
                st.session_state["bi_next_id"] = pid + 1
                bi_plots.append({"id": pid, "type": add_viz, "width": "full"})
                st.rerun()
        with bd:
            if st.button("Add plot (half-width)", key="bi_add_btn_half", icon=":material/add:"):
                pid = st.session_state.get("bi_next_id", 0)
                st.session_state["bi_next_id"] = pid + 1
                bi_plots.append({"id": pid, "type": add_viz, "width": "half"})
                st.rerun()

    if not bi_plots:
        st.caption("Use the button above to add a plot.")

    # ── Render each plot card ─────────────────────────────────────────────
    theme = _plot_theme()

    leftover_col = None

    for i, pspec in enumerate(list(bi_plots)):
        pid = pspec["id"]

        if pspec["width"] == "half":
            if leftover_col is None:
                hc1, hc2 = st.columns([1, 1])
                plot_col = hc1
                leftover_col = hc2
            else:
                plot_col = leftover_col
                leftover_col = None
        else:
            plot_col = st.container()

        with plot_col:
            with st.container(border=True):
                hc1, hc2 = st.columns([3, 1])
                with hc1:
                    viz_type = st.selectbox(
                        "Visualisation", VIZ_TYPES,
                        index=VIZ_TYPES.index(pspec["type"]),
                        key=f"bi_{pid}_type",
                        label_visibility="collapsed",
                    )
                with hc2:
                    if st.button("Remove", key=f"bi_rm_{pid}", icon=":material/delete:"):
                        bi_plots.pop(i)
                        st.rerun()

                with st.expander("Show/hide plot", expanded=True):

                    if viz_type == "Cartoon binary evolution":
                        try:
                            fig, _ = pop.plot_cartoon_binary(bin_num=selected_bn)
                            show_fig(fig)
                        except Exception as e:
                            st.error(f"Could not plot cartoon: {e}")

                    elif viz_type == "Detailed binary evolution":
                        if not hasattr(pop, "bcm") or pop.bcm is None:
                            st.warning("Detailed binary evolution data not available. "
                                    "Re-run with 'Store full BCM' enabled.")
                        else:
                            try:
                                bcm_bn   = _downsample_bcm(pop.bcm.loc[selected_bn])
                                time_col = "tphys"
                                if time_col in bcm_bn.columns and len(bcm_bn) > 1:
                                    t_lo_myr = float(bcm_bn[time_col].min())
                                    t_hi_myr = float(bcm_bn[time_col].max())
                                    t_lo_yr_log10 = np.log10(max(t_lo_myr * 1e6, 1e5))
                                    t_hi_yr_log10 = np.log10(t_hi_myr * 1e6)
                                    t_range = st.slider(
                                        "Time range (log(tphys/yr))",
                                        min_value=round(t_lo_yr_log10, 4),
                                        max_value=round(t_hi_yr_log10, 4),
                                        value=(round(t_lo_yr_log10, 4), round(t_hi_yr_log10, 4)),
                                        step=0.01,
                                        key=f"bi_{pid}_evol_range",
                                    )
                                    lo_myr = 10 ** t_range[0] / 1e6
                                    hi_myr = 10 ** t_range[1] / 1e6
                                    bcm_plot = bcm_bn[
                                        (bcm_bn[time_col] >= lo_myr) & (bcm_bn[time_col] <= hi_myr)
                                    ]
                                    if bcm_plot.empty:
                                        st.warning("No data in selected time range.")
                                        bcm_plot = bcm_bn
                                else:
                                    bcm_plot = bcm_bn
                                plt.style.use("default")
                                fig = cosmic.plotting.plot_binary_evol(bcm_plot)
                                show_fig(fig)
                            except Exception as e:
                                st.error(f"Could not plot detailed evolution: {e}")

                    elif viz_type == "Galactic orbit":
                        if not has_orbits:
                            st.warning("Galactic orbits were not evolved. "
                                    "Re-run with 'Evolve galactic orbits' enabled.")
                        else:
                            bn_idx = list(pop.bin_nums).index(selected_bn)
                            tau_yr = float(pop._initial_galaxy._tau[bn_idx].to(u.yr).value)
                            orb_range = st.slider(
                                "Time range (log(t/yr))",
                                min_value=5.0,
                                max_value=round(np.log10(tau_yr), 4),
                                value=(5.0, round(np.log10(tau_yr), 4)),
                                step=0.01,
                                key=f"bi_{pid}_orb_range",
                            )
                            t_min_yr = 10 ** orb_range[0]
                            t_max_yr = 10 ** orb_range[1]
                            try:
                                fig = _plot_orbit_plotly(pop, selected_bn, t_min_yr, t_max_yr,
                                                        theme, animated=False)
                                st.plotly_chart(fig, width='stretch')
                            except Exception as e:
                                st.error(f"Could not plot orbit: {e}")

                            st.markdown("**Animation**")
                            st.caption("Visualise how the orbit evolves over time. "
                                    "Uses the same time range as the plot above.")
                            col_play, _ = st.columns([1, 4])
                            with col_play:
                                anim_key = f"bi_{pid}_orb_anim"
                                if st.button("Generate animation", key=f"{anim_key}_btn"):
                                    st.session_state[anim_key] = True
                                if st.session_state.get(anim_key) and st.button(
                                        "Clear animation", key=f"{anim_key}_clear"):
                                    st.session_state[anim_key] = False
                            if st.session_state.get(anim_key):
                                try:
                                    anim_fig = _plot_orbit_plotly(
                                        pop, selected_bn, t_min_yr, t_max_yr,
                                        theme, animated=True,
                                    )
                                    st.plotly_chart(anim_fig, width='stretch')
                                except Exception as e:
                                    st.error(f"Could not generate animation: {e}")

                    elif viz_type == "Hertzsprung-Russell Diagram":
                        if not hasattr(pop, "bcm") or pop.bcm is None:
                            st.warning("BCM data not available. Re-run with 'Store full BCM' enabled.")
                        else:
                            try:
                                fig = _plot_hrd_plotly(pop, selected_bn, theme)
                                st.plotly_chart(fig, width='stretch')
                            except Exception as e:
                                st.error(f"Could not plot HR diagram: {e}")


# ──────────────────────────────────────────────
# Code generation
# ──────────────────────────────────────────────

_UNIT_CODE = {
    "Gyr": "u.Gyr", "Myr": "u.Myr", "yr": "u.yr",
    "kpc": "u.kpc", "pc":  "u.pc",
    "solMass": "u.Msun",
    "km / s": "u.km / u.s",
    "1 / kpc": "1 / u.kpc",
}


def _qty(q):
    unit_s = str(q.unit)
    unit_c = _UNIT_CODE.get(unit_s, f"u.Unit('{unit_s}')")
    v = q.value
    if hasattr(v, "item"):
        v = v.item()
    return f"{v if v != int(v) else int(v)} * {unit_c}"


def _sfh_params_code(d: dict) -> str:
    """Return a single-line dict literal for sfh_params."""
    if not d:
        return "{}"
    items = []
    for k, v in d.items():
        if isinstance(v, u.Quantity):
            items.append(f'"{k}": {_qty(v)}')
        else:
            items.append(f'"{k}": {v!r}')
    return "{" + ", ".join(items) + "}"


def _bse_settings_code(binfrac: float, bse_overrides: dict) -> str:
    """Return a dict literal for BSE_settings."""
    merged = {"binfrac": binfrac, **bse_overrides}
    items = [f'"{k}": {v!r}' for k, v in merged.items()]
    return "{" + ", ".join(items) + "}"


def generate_code(indent=" " * 4) -> str:
    job_type = st.session_state.get("last_job_type", "")
    params   = st.session_state.get("last_job_params", {})
    if not job_type or not params:
        return "# No simulation has been run yet."

    pot_name    = params.get("potential_name")
    pot_params  = params.get("potential_params")
    pot_code    = make_potential_code(pot_name, pot_params)
    needs_galactic = "galactic" in pot_code
    bse_overrides  = params.get("bse_overrides")

    # ── visualisation lines ────────────────────────────────────────────────)
    bi_plots = st.session_state.get("bi_plots", [])
    bn = st.session_state.get("bi_bn", 0)
    bi_lines = []
    for bi_viz in bi_plots:
        plot_type = st.session_state.get(f"bi_{bi_viz['id']}_type")
        if bi_lines != []:
            bi_lines.append("")
        if plot_type == "Cartoon binary evolution" and bn is not None:
            bi_lines += [
                "# cartoon binary evolution",
                f"pop.plot_cartoon_binary(bin_num={bn})",
            ]
        elif plot_type == "Galactic orbit" and bn is not None:
            # get the value of the slider with the right id
            id = bi_viz["id"]
            t_range = st.session_state.get(f"bi_{id}_orb_range")
            if t_range is not None:
                t_min_gyr = round(10 ** t_range[0] / 1e9, 3)
                t_max_gyr = round(10 ** t_range[1] / 1e9, 3)
                t_args = f", t_min={t_min_gyr} * u.Gyr, t_max={t_max_gyr} * u.Gyr"
            else:
                t_args = ""
            bi_lines += [
                "# galactic orbit evolution" if t_range is None else f"# galactic orbit evolution (between {t_min_gyr} - {t_max_gyr} Gyr)",
                f"pop.plot_orbit(bin_num={bn}{t_args})",
            ]
        elif plot_type == "Hertzsprung-Russell Diagram" and bn is not None:
            bi_lines += [
                "# HR diagram",
                f"pop.plot_hrd(bin_num={bn})",
            ]
        elif plot_type == "Detailed binary evolution" and bn is not None:
            bi_lines += [
                "# detailed binary evolution",
                f"bcm_bn = pop.bcm.loc[{bn}]",
                "with plt.style.context('default'):",
                f"{indent}fig = cosmic.plotting.plot_binary_evol(bcm_bn)",
                "plt.show()",
            ]
        else:
            st.warning(f"Could not generate code for plot type '{plot_type}' (binary number {bn}).")

    pop_plot_lines = []
    pop_plots = st.session_state.get("pop_plots", [])
    for pop_plot in pop_plots:
        plot_id = pop_plot["id"]
        plot_type = st.session_state.get(f"pop_plot_{plot_id}_type")
        plot_table = st.session_state.get(f"pop_plot_{plot_id}_table")
        xcol = st.session_state.get(f"pop_plot_{plot_id}_xcol")
        ycol = st.session_state.get(f"pop_plot_{plot_id}_ycol")
        ccol = st.session_state.get(f"pop_plot_{plot_id}_ccol")
        
        if pop_plot_lines != []:
            pop_plot_lines.append("")  # add a blank line between plots
        if plot_type == "Histogram":
            log_x = st.session_state.get(f"pop_plot_{plot_id}_hlogx")
            log_y = st.session_state.get(f"pop_plot_{plot_id}_hlogy")
            nbins = st.session_state.get(f"pop_plot_{plot_id}_hbins")
            xdata = (
                f"np.log10(pop.{plot_table}[{xcol!r}])"
                if log_x else
                f"pop.{plot_table}[{xcol!r}]"
            )
            pop_plot_lines += [
                "fig, ax = plt.subplots()",
                f"ax.hist({xdata}, bins={nbins})",
            ]
            if log_y:
                pop_plot_lines.append("ax.set_yscale('log')")
            xlabel = f"'log10({xcol})'" if log_x else repr(xcol)
            pop_plot_lines += [f"ax.set_xlabel({xlabel})", "ax.set_ylabel('Count')", "plt.show()"]
        else:  # Scatter
            log_x = st.session_state.get(f"pop_plot_{plot_id}_slogx")
            log_y = st.session_state.get(f"pop_plot_{plot_id}_slogy")
            color_arg = f", c=pop.{plot_table}[{ccol!r}]" if ccol != "(none)" else ""
            pop_plot_lines += [
                "fig, ax = plt.subplots()",
                f"sc = ax.scatter(pop.{plot_table}[{xcol!r}], pop.{plot_table}[{ycol!r}]{color_arg})",
            ]
            if ccol != "(none)":
                pop_plot_lines.append(f"fig.colorbar(sc, ax=ax, label={ccol!r})")
            if log_x:
                pop_plot_lines.append("ax.set_xscale('log')")
            if log_y:
                pop_plot_lines.append("ax.set_yscale('log')")
            pop_plot_lines += [f"ax.set_xlabel({xcol!r})", f"ax.set_ylabel({ycol!r})", "plt.show()"]

    # ── header ────────────────────────────────────────────────────────────
    lines = [
        "import cogsworth",
        "import gala.potential as gp",
    ]
    if needs_galactic:
        lines.append("from gala.units import galactic")
    lines += [
        "from astropy import units as u",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        f"pot = {pot_code}",
        "",
    ]

    # ── simulation body ───────────────────────────────────────────────────
    if job_type == "population":
        sfh_name = params.get("sfh_name", "Wagg+2022 (default)")
        sfh_cls  = SFH_CLASS_CODE.get(sfh_name, "sfh.Wagg2022")
        sfh_p    = params.get("sfh_params_dict", {})
        n        = params.get("n_binaries", 100)
        t_max    = params.get("max_ev_time", 12.0)
        fk1      = params.get("final_kstar1", list(range(16)))
        fk2      = params.get("final_kstar2", list(range(16)))
        v_disp   = params.get("v_dispersion", 5.0)
        orbits   = params.get("evolve_orbits", True)
        samp_params = params.get("sampling_params", {})
        bfm = samp_params.get("binfrac_model", 0.5)
        binfrac_bse = bfm if isinstance(bfm, (int, float)) else 0.5
        bse_code = _bse_settings_code(binfrac_bse, bse_overrides)

        full_bcm = params.get("full_bcm", False)
        bcm_line = '    bcm_timestep_conditions=[["dtp=0.0"]],' if full_bcm else ""
        samp_line = f"{indent}sampling_params={samp_params!r}," if samp_params else ""
        lines += [
            "pop = cogsworth.pop.Population(",
            f"{indent}n_binaries={n},",
            f"{indent}sfh_model=cogsworth.{sfh_cls},",
            f"{indent}sfh_params={_sfh_params_code(sfh_p)},",
            f"{indent}galactic_potential=pot,",
            f"{indent}max_ev_time={t_max} * u.Gyr,",
            f"{indent}final_kstar1={fk1},",
            f"{indent}final_kstar2={fk2},",
            f"{indent}v_dispersion={v_disp} * u.km / u.s,",
            f"{indent}store_entire_orbits={orbits},",
            f"{indent}processes=2,",
            f"{indent}BSE_settings={bse_code},",
            f"{indent}use_default_BSE_settings=True,",
            *(([samp_line]) if samp_line else []),
            *(([bcm_line]) if bcm_line else []),
            ")",
            "pop.sample_initial_binaries()",
            "pop.perform_stellar_evolution()",
            "pop.perform_galactic_evolution()" if orbits else "# pop.perform_galactic_evolution()  # skipped",
            "",
        ]

    else:  # single_binary
        m1     = params.get("mass_1", 10.0)
        q      = params.get("mass_ratio", 0.8)
        m2     = round(m1 * q, 4)
        porb   = params.get("porb", 100.0)
        ecc    = params.get("ecc", 0.0)
        Z      = params.get("metallicity", 0.014)
        tau    = params.get("tau", 5.0)
        x, y, z = params.get("x", 8.0), params.get("y", 0.0), params.get("z", 0.0)
        vR, vT, vz = params.get("v_R", 0.0), params.get("v_T", 220.0), params.get("v_z", 0.0)
        orbits = params.get("evolve_orbits", True)
        bse_code = _bse_settings_code(1.0, bse_overrides)

        # add InitialBinaryTable import to header
        lines.insert(1, "from cosmic.sample import InitialBinaryTable")

        lines += [
            "# construct initial binary using COSMIC",
            f"ib = InitialBinaryTable.InitialBinaries(",
            f"{indent}m1={m1}, m2={m2}, porb={porb}, ecc={ecc},",
            f"{indent}tphysf={tau * 1000:.1f},",
            f"{indent}kstar1=1, kstar2=1, metallicity={Z},",
            ")",
            "",
            "# build single item StarFormationHistory",
            "ig = cogsworth.sfh.StarFormationHistory(1, immediately_sample=False)",
            f"ig._tau = np.array([{tau}]) * u.Gyr",
            f"ig._Z = np.array([{Z}])",
            f"ig._x = np.array([{x}]) * u.kpc",
            f"ig._y = np.array([{y}]) * u.kpc",
            f"ig._z = np.array([{z}]) * u.kpc",
            'ig._which_comp = np.array(["thin_disc"])',
            f"ig.v_R = np.array([{vR}]) * u.km / u.s",
            f"ig.v_T = np.array([{vT}]) * u.km / u.s",
            f"ig.v_z = np.array([{vz}]) * u.km / u.s",
            "",
            "# initialise the population",
            "pop = cogsworth.pop.Population(",
            f"{indent}n_binaries=1,",
            f"{indent}sfh_model=cogsworth.sfh.StarFormationHistory,",
            f"{indent}galactic_potential=pot,",
            f"{indent}max_ev_time={tau} * u.Gyr,",
            f"{indent}final_kstar1=list(range(16)),",
            f"{indent}final_kstar2=list(range(16)),",
            f"{indent}store_entire_orbits={orbits},",
            f"{indent}processes=1,",
            f"{indent}BSE_settings={bse_code},",
            f"{indent}use_default_BSE_settings=True,",
            f"{indent}bcm_default_timestep=0.0,",
            ")",
            ""
            "# set initial conditions directly without sampling",
            "pop._initial_binaries = ib",
            "pop._initial_galaxy   = ig",
            "",
            "# evolve the binary",
            "pop.perform_stellar_evolution()",
            "pop.perform_galactic_evolution()" if orbits else "# pop.perform_galactic_evolution()  # skipped",
            "",
        ]

    if bi_lines:
        lines += [
            "# -------------------------------",
            "# --- individual binary plots ---",
            "# -------------------------------",
            ""
        ]
        lines += bi_lines
    
    if pop_plot_lines:
        if bi_lines:
            lines.append("")
        lines += [
            "# ------------------------------",
            "# --- population level plots ---",
            "# ------------------------------",
        ]
        lines += pop_plot_lines
    return "\n".join(lines)


# with st.sidebar:
#     # links to documentation and github
#     st.markdown("## Resources")
    
#     st.link_button(
#         "cogsworth documentation",
#         "https://cogsworth.readthedocs.io/",
#         icon=":material/book_2:"
#     )

#     st.link_button(
#         "GitHub Repo",
#         "https://github.com/TomWagg/cogsworth",
#         icon=":material/code:"
#     )




# ──────────────────────────────────────────────
# Input tabs
# ──────────────────────────────────────────────

tab_pop, tab_single = st.tabs(["Full population", "One binary"])

# ── Tab 1: Population Mode ──────────────────
with tab_pop:
    st.subheader("Run a population of binaries")
    st.markdown("In this mode, you can run a full population of binaries. Use the inputs below to set up your simulation (this is equivalent to initializing a `cogsworth.pop.Population` object with your chosen parameters).")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_binaries   = st.slider("Number of systems", 1, 1_000, 100, key="pop_n", help="Number of systems (binaries/singles) to evolve.")
        sfh_name     = st.selectbox("Star-formation history model", SFH_MODEL_NAMES, key="pop_sfh", help="Choose a star-formation history (SFH) model for the birth times,  positions, and metallicity of the binaries. Each model has its own parameters, which you can set in the expander that appears after you select the model.")
        sfh_p = sfh_params_form(sfh_name)
        max_ev_time  = st.slider("Max evolution time (Gyr)", 0.0, 14.0, 12.0, 0.25, key="pop_tmax", help="Maximum evolution time. This is the maximum forward evolution time for each binary. Binaries will be evolved from their birth time to this time.")
    with col2:
        potential_name = st.selectbox("Galactic potential", POTENTIAL_NAMES, key="pop_pot", help="Choose a galactic potential model. This potential will be used for orbit integration. Each potential has its own parameters, which you can set in the expander that appears after you select the model. For more information, check out the [gala documentation](https://gala.adrian.pw/en/latest/potential/index.html).")
        pot_p = potential_params_form(potential_name, key_prefix="pop")
        v_dispersion   = st.number_input("Velocity dispersion (km/s)", 0.0, 100.0, 5.0, key="pop_vdisp", help="3D velocity dispersion to apply to initial velocities of binaries. For star formation histories that don't provide a velocity, the velocity will be set to the circular velocity at its birth position plus a random component drawn from a Gaussian with this velocity dispersion. For star formation histories that do provide a velocity, this dispersion is ignored.")
        evolve_orbits = st.checkbox("Evolve galactic orbits", value=True, key="pop_orbits", help="Whether to perform galactic orbit integration. If unchecked, the binaries will be evolved in place without any galactic dynamics. Note that orbit integration increases the runtime, especially for large populations and complex potentials.")
        full_bcm = st.checkbox("Store full BCM", value=False, key="pop_full_bcm", help="Store all binary timestep evolution data (bcm table). This enables the 'Detailed binary evolution' plot for any binary in the population, but significantly increases memory usage and runtime. Only enable this when you plan to inspect specific binaries in detail.")
    with col3:
        fk1 = st.multiselect("Target final stellar type 1 (primary)", KSTAR_OPTIONS, default=[], key="pop_fk1",
                             help="Select which final stellar types to target for the primary star (the initially more massive star) in the binary. This will change the way in which the IMF is sampled and prioritise binaries that are likely to produce these types. By default, no preferential sampling occurs. Note that cogsworth stores the total mass necessary to produce the population, so you can re-normalise results.")
        fk1 = [KSTAR_TO_INT[s] for s in (fk1 if fk1 != [] else KSTAR_OPTIONS)]
        fk2 = st.multiselect("Target final stellar type 2 (secondary)", KSTAR_OPTIONS, default=[], key="pop_fk2",
                             help="Select which final stellar types to target for the secondary star (the initially less massive star) in the binary. This will change the way in which the IMF is sampled and prioritise binaries that are likely to produce these types. By default, no preferential sampling occurs. Note that cogsworth stores the total mass necessary to produce the population, so you can re-normalise results.")
        fk2 = [KSTAR_TO_INT[s] for s in (fk2 if fk2 != [] else KSTAR_OPTIONS)]
        samp_p  = sampling_params_form(key_prefix="pop_samp")
        if n_binaries > 2000 and evolve_orbits:
            st.warning("Orbit integration for >2000 binaries may take several minutes.")

    colp_1, colp_2 = st.columns(2)
    with colp_1:
        bse_ovr = bse_settings_form(key_prefix="pop_bse")

    if st.button("Evolve population!", type="primary", key="btn_pop", width='stretch'):
        if not fk1 or not fk2:
            st.error("Select at least one stellar type for each final_kstar filter.")
        else:
            clear_job()
            params = dict(
                sfh_name=sfh_name,
                sfh_params_dict=sfh_p,
                n_binaries=n_binaries,
                max_ev_time=float(max_ev_time),
                final_kstar1=fk1,
                final_kstar2=fk2,
                potential_name=potential_name,
                potential_params=pot_p,
                v_dispersion=float(v_dispersion),
                evolve_orbits=evolve_orbits,
                bse_overrides=dict(bse_ovr),
                full_bcm=full_bcm,
                sampling_params=dict(samp_p),
            )
            st.session_state.update({
                "last_job_type":  "population",
                "last_job_params": params,
                "max_ev_time":    float(max_ev_time),
                "pop_has_orbits": evolve_orbits,
                "job_id": queue_db.submit_job("population", params),
            })
            st.rerun()


# ── Tab 2: Single Binary Mode ───────────────
with tab_single:
    st.subheader("Run one binary")
    st.markdown("In this mode, you can run a single binary with specified initial parameters. Use the inputs below to set up your simulation (this is equivalent to initializing a `cogsworth.pop.Population` object with `n_binaries=1` and providing an `InitialBinaryTable` and `StarFormationHistory` with your chosen parameters).")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Stellar parameters**")
        colm1, colm2 = st.columns(2)
        with colm1:
            mass_1     = st.number_input(r"Primary mass, $m_1 \, [\rm M_\odot]$", 0.5, 150.0, 20.0, key="sb_m1", help="Mass of the primary star (the initially more massive star) in solar masses.")
        with colm2:
            mass_ratio = st.slider("Mass ratio, $q = m_2/m_1$", 0.01, 1.0, 0.8, 0.01, key="sb_q", help="Mass ratio of the binary, defined as the mass of the secondary star (the initially less massive star) divided by the mass of the primary star. This must be between 0 and 1. The mass of the secondary star will be calculated as $m_2 = q \\, m_1$.")
            st.caption(f"$m_2 = {mass_ratio * mass_1:.2f} \, M_\odot$")
        
        colpe1, colpe2 = st.columns(2)
        with colpe1:
            porb = st.number_input(r"Orbital period, $P \, [\rm days]$", 0.01, 1e10, 10000.0, key="sb_porb", help="Orbital period of the binary in days.")
        with colpe2:
            ecc         = st.slider("Eccentricity, $e$", 0.0, 0.99, 0.0, 0.01, key="sb_ecc", help="Eccentricity of the binary orbit.")
        metallicity = st.number_input(r"Metallicity, $Z$", 1e-4, 0.03, 0.014, format="%.4f", key="sb_Z", help="Metallicity of the binary system.")
    with col2:
        st.markdown("**Galactic parameters**")
        tau   = st.slider(r"Lookback time, $\tau \, [\rm Gyr]$", 0.01, 14.0, 5.0, 0.1, key="sb_tau", help="Lookback time to the birth of the binary system. The binary will be evolved forward from this time to the present day.")
        colpos1, colpos2, colpos3 = st.columns(3)
        with colpos1:
            x_b = st.number_input(r"$x \, [\rm kpc]$", -30.0, 30.0, 8.0, key="sb_x", help="Birth x-coordinate of the binary in kiloparsecs (galactocentric).")
        with colpos2:
            y_b = st.number_input(r"$y \, [\rm kpc]$", -30.0, 30.0, 0.0, key="sb_y", help="Birth y-coordinate of the binary in kiloparsecs (galactocentric).")
        with colpos3:
            z_b = st.number_input(r"$z \, [\rm kpc]$", -5.0, 5.0, 0.0, key="sb_z", help="Birth z-coordinate of the binary in kiloparsecs (galactocentric).")

        colvel1, colvel2, colvel3 = st.columns(3)
        with colvel1:
            v_R_b = st.number_input(r"$v_R \, [\rm km/s]$", -300.0, 300.0, 0.0, key="sb_vR", help="Radial velocity of the binary in km/s.")
        with colvel2:
            v_T_b = st.number_input(r"$v_T \, [\rm km/s]$", -500.0, 500.0, 220.0, key="sb_vT", help="Tangential velocity of the binary in km/s.")
        with colvel3:
            v_z_b = st.number_input(r"$v_z \, [\rm km/s]$", -300.0, 300.0, 0.0, key="sb_vz", help="Vertical velocity of the binary in km/s.")

    st.markdown("**Settings**")

    col3, col4 = st.columns(2)
    with col3:
        sb_potential = st.selectbox("Galactic potential", POTENTIAL_NAMES, key="sb_pot", help="Choose a galactic potential model. This potential will be used for orbit integration. Each potential has its own parameters, which you can set in the expander that appears after you select the model. For more information, check out the [gala documentation](https://gala.adrian.pw/en/latest/potential/index.html).")
        sb_orbits = st.checkbox("Evolve galactic orbit", value=True, key="sb_orbits", help="Whether to perform galactic orbit integration. If unchecked, the binary will be evolved in place without any galactic dynamics. Note that orbit integration increases the runtime, especially for complex potentials.")
    with col4:
        sb_pot_p   = potential_params_form(sb_potential, key_prefix="sb")
        sb_bse_ovr = bse_settings_form(key_prefix="sb_bse")

    if st.button("Evolve binary!", type="primary", key="btn_single", width='stretch'):
        clear_job()
        params = dict(
            mass_1=float(mass_1), mass_ratio=float(mass_ratio),
            porb=float(porb), ecc=float(ecc), metallicity=float(metallicity),
            x=float(x_b), y=float(y_b), z=float(z_b),
            v_R=float(v_R_b), v_T=float(v_T_b), v_z=float(v_z_b),
            tau=float(tau), potential_name=sb_potential,
            potential_params=sb_pot_p,
            evolve_orbits=sb_orbits,
            bse_overrides=dict(sb_bse_ovr),
        )
        st.session_state.update({
            "last_job_type":   "single_binary",
            "last_job_params":  params,
            "max_ev_time":     float(tau),
            "pop_has_orbits":  sb_orbits,
            "job_id": queue_db.submit_job("single_binary", params),
        })
        st.rerun()


# ──────────────────────────────────────────────
# Job status (polls until done)
# ──────────────────────────────────────────────

if "job_id" in st.session_state and "pop" not in st.session_state:
    job_id = st.session_state["job_id"]
    job    = queue_db.get_job(job_id)

    if job is None:
        st.error("Job record not found. The database may have been cleared.")
        if st.button("Dismiss", key="dismiss_missing"):
            clear_job(); st.rerun()
        st.stop()

    status = job["status"]
    st.divider()

    if status == "pending":
        msg_box  = st.empty()
        warn_box = st.empty()
        while True:
            job = queue_db.get_job(job_id)
            if job["status"] != "pending":
                break
            pos = queue_db.get_queue_position(job_id)
            age = time.time() - job["created_at"]
            msg = ("You're next in the queue — waiting for the current job to finish..."
                   if pos == 0 else f"{pos} job(s) ahead of you in the queue.")
            msg_box.info(msg)
            if age > WORKER_WARN_SECS:
                warn_box.warning("Job has been queued for a while — is the worker running? "
                                 "Start it with `bash webapp/run.sh`.")
            time.sleep(2)
        st.rerun()

    elif status == "running":
        with st.spinner("Evolving — this may take a few minutes..."):
            while True:
                time.sleep(2)
                job = queue_db.get_job(job_id)
                if job["status"] != "running":
                    break
        st.rerun()

    elif status == "done":
        with st.spinner("Loading results..."):
            try:
                parts = ["initial_binaries", "initial_galaxy", "stellar_evolution"]
                if st.session_state.get("pop_has_orbits", False):
                    parts.append("galactic_orbits")
                pop = cogsworth.pop.load(job["result_path"], parts=parts)
                st.session_state["pop"] = pop
                # Cache bytes now while the file still exists
                with open(job["result_path"], "rb") as fh:
                    st.session_state["pop_h5_bytes"] = fh.read()
            except Exception as e:
                st.error(f"Failed to load results: {e}")
                if st.button("Clear", key="clear_load_err"):
                    clear_job(); st.rerun()
                st.stop()
        st.rerun()

    elif status == "failed":
        st.error("Evolution failed.")
        with st.expander("Error details"):
            st.code(job["error_msg"])
        if st.button("Clear and try again", key="clear_failed"):
            clear_job(); st.rerun()
        st.stop()

    else:
        st.stop()


# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────

if "pop" not in st.session_state:
    st.stop()

pop: cogsworth.pop.Population = st.session_state["pop"]
has_orbits: bool = st.session_state.get("pop_has_orbits", False)
max_ev_time_stored: float = st.session_state.get("max_ev_time", 12.0)
is_single: bool = st.session_state.get("last_job_type") == "single_binary"

st.divider()
st.header("Simulation results")

# Single binary: skip Data Explorer, go straight to Binary Inspector
if is_single:
    _render_binary_inspector(pop, has_orbits)

else:
    res_tab_data, res_tab_inspect = st.tabs(["Explore the full population", "Inspect a specific binary"])

    # ── Data Explorer ─────────────────────────
    with res_tab_data:

        # ── Add-plot toolbar ──────────────────────────────────────────────
        pop_plots = st.session_state.setdefault("pop_plots", [])
        ca, cb, _ = st.columns([2, 1, 5])
        with ca:
            add_type = st.selectbox("", ["Histogram", "Scatter"],
                                    key="de_add_type", label_visibility="collapsed")
        with cb:
            if st.button("Add plot", key="de_add_btn", icon=':material/add:'):
                pid = st.session_state.get("de_next_id", 0)
                st.session_state["de_next_id"] = pid + 1
                pop_plots.append({"id": pid, "type": add_type})
                st.rerun()

        if not pop_plots:
            st.caption("Use the button above to add a plot.")

        # ── Render each plot card ─────────────────────────────────────────
        for i, pspec in enumerate(list(pop_plots)):
            pid = pspec["id"]

            with st.container(border=True):

                with st.expander("Plot settings", expanded=True):
                    hc1, hc2, hc3 = st.columns([3, 2, 1])
                    with hc1:
                        table_choice = st.segmented_control(
                            "Table", ["final_bpp", "initial_binaries", "kick_info"],
                            key=f"pop_plot_{pid}_table",
                            default=pspec.get("table_choice", "final_bpp"),
                        )
                    with hc2:
                        plot_type = st.segmented_control(
                            "Type", ["Histogram", "Scatter"],
                            key=f"pop_plot_{pid}_type",
                            default=pspec["type"],
                        )

                    df = getattr(pop, table_choice)
                    if df is None or df.empty:
                        st.warning("No data available for this table.")
                        continue
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

                    if plot_type == "Histogram":
                        col_h, col_opts = st.columns([1, 3])
                        with col_h:
                            xcol = st.selectbox("Column", numeric_cols, key=f"pop_plot_{pid}_xcol")
                        with col_opts:
                            log_x = st.checkbox("Log x-axis", key=f"pop_plot_{pid}_hlogx")
                            log_y = st.checkbox("Log y-axis", key=f"pop_plot_{pid}_hlogy")
                            nbins = st.slider("Bins", 5, 200, 50, key=f"pop_plot_{pid}_hbins")

                        data = df[xcol].replace([np.inf, -np.inf], np.nan).dropna()
                        if log_x and (data > 0).any():
                            data    = np.log10(data[data > 0])
                            x_label = f"log10({xcol})"
                        else:
                            x_label = xcol

                    else:  # Scatter
                        col_s1, col_s2, col_s3, col_opts = st.columns([2, 2, 2, 1])
                        with col_s1:
                            xcol = st.selectbox("x-axis", numeric_cols, index=0,
                                                key=f"pop_plot_{pid}_xcol")
                        with col_s2:
                            ycol = st.selectbox("y-axis", numeric_cols,
                                                index=min(1, len(numeric_cols) - 1),
                                                key=f"pop_plot_{pid}_ycol")
                        with col_s3:
                            color_opts_list = ["(none)"] + numeric_cols
                            color_col = st.selectbox("Colour by", color_opts_list,
                                                    key=f"pop_plot_{pid}_ccol")
                        with col_opts:
                            log_x = st.checkbox("Log x", key=f"pop_plot_{pid}_slogx")
                            log_y = st.checkbox("Log y", key=f"pop_plot_{pid}_slogy")
                            log_c = st.checkbox("Log colour", key=f"pop_plot_{pid}_slogc")

                table_label = "Present day" if table_choice == "final_bpp" else "Initial state"
                if plot_type == "Histogram":
                    fig = px.histogram(data, nbins=nbins, log_y=log_y,
                                    labels={"value": x_label, "count": "Count"},
                                    title=f"Distribution of {xcol}  [{table_label}]",
                                    color_discrete_sequence=["#905cc4"])
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch', key=f"hist_{pid}")

                else:
                    plot_df = df[
                        [xcol, ycol] + ([color_col] if color_col != "(none)" else [])
                    ].replace([np.inf, -np.inf], np.nan).dropna()

                    c_kwarg = {}
                    if color_col != "(none)":
                        if log_c and (plot_df[color_col] > 0).any():
                            plot_df = plot_df.copy()
                            plot_df[color_col] = np.log10(plot_df[color_col].clip(lower=1e-30))
                            c_kwarg = {"color": color_col, "color_continuous_scale": "viridis",
                                        "labels": {color_col: f"log10({color_col})"}}
                        else:
                            c_kwarg = {"color": color_col, "color_continuous_scale": "viridis"}
                    else:
                        c_kwarg = {"color_discrete_sequence": ["#905cc4"]}

                    fig = px.scatter(plot_df, x=xcol, y=ycol,
                                        log_x=log_x, log_y=log_y,
                                        title=f"{ycol} vs {xcol} [{table_label}]",
                                        opacity=0.6, **c_kwarg)
                    st.plotly_chart(fig, width='stretch', key=f"scatter_{pid}")

                if st.button("Remove", key=f"de_rm_{pid}", icon=":material/delete:", width='stretch'):
                    pop_plots.pop(i)
                    st.rerun()

        with st.expander("View raw tables"):
            col_tbl, col_bns = st.columns([1, 1])
            with col_tbl:
                raw_tbls = st.pills("Which table do you want to see?", ["bpp", "final_bpp", "initial_binaries", "kick_info"], selection_mode="multi", key="de_raw_table")
                raw_dfs = [getattr(pop, tbl) for tbl in raw_tbls]
            with col_bns:
                raw_bns = st.text_input("Which binaries do you want in the table?", placeholder="Comma-separated list, e.g. 1,3,10-12,24 [leave blank for all binaries]")
                if raw_bns.strip():
                    try:
                        bins_to_show = set()
                        for part in raw_bns.split(","):
                            if "-" in part:
                                start, end = map(int, part.split("-"))
                                bins_to_show.update(range(start, end + 1))
                            else:
                                bins_to_show.add(int(part))
                        raw_dfs = [df[df["bin_num"].isin(bins_to_show)] for df in raw_dfs]
                    except Exception as e:
                        st.warning(f"Failed to parse binary numbers: {e}")
                        raw_dfs = [None] * len(raw_dfs)

            if st.button("Translate column names", key="de_translate", help="Converts COSMIC stellar types and evol_types codes to translated labels"):
                try:
                    pop.translate_tables()
                    st.session_state["pop"] = pop
                    st.success("Columns translated.")
                    st.rerun()
                except Exception as e:
                    st.warning(f"Translation failed: {e}")
            if any(df is not None for df in raw_dfs):
                for tbl, df in zip(raw_tbls, raw_dfs):
                    if df is not None:
                        st.markdown(f"**{tbl}**")
                        st.dataframe(df, width='stretch')

    # ── Binary Inspector ──────────────────────
    with res_tab_inspect:
        _render_binary_inspector(pop, has_orbits)

# ── Download button ──────────────────────────
h5_bytes = st.session_state.get("pop_h5_bytes")
if h5_bytes:
    st.download_button(
        "Download population (.h5)",
        data=h5_bytes,
        icon=':material/download:',
        file_name="cogsworth_online_population.h5",
        mime="application/x-hdf5",
        key="dl_pop",
    )


# ──────────────────────────────────────────────
# Code generation (bottom of page)
# ──────────────────────────────────────────────

st.divider()
with st.expander("Reproduce this simulation in Python", expanded=False):
    st.markdown(
        "The code below reproduces the current simulation and the selected visualisations. "
        "You can download the .py file or just copy it into your own file/notebook."
    )
    code = generate_code()
    st.code(code, language="python")
    st.download_button(
        "Download as .py",
        data=code,
        file_name="cogsworth_simulation.py",
        mime="text/plain",
        key="dl_code",
    )
