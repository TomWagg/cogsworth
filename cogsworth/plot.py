import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from .utils import kstar_translator, evol_type_translator

__all__ = ["plot_cmd", "plot_cartoon_evolution", "plot_galactic_orbit"]


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


def plot_cmd(pop, m_filter="Gaia_G_EDR3", c_filter_1="Gaia_BP_EDR3", c_filter_2="Gaia_RP_EDR3",
             fig=None, ax=None, show=True, **kwargs):
    """Plot a colour-magnitude diagram for a population.

    Parameters
    ----------
    pop : :class:`~cogsworth.pop.Population`
        Population to plot.
    m_filter : `str`, optional
        Filter for the magnitude, by default "G"
    c_filter_1 : `str`, optional
        Filter for the first colour, by default "BP"
    c_filter_2 : `str`, optional
        Filter for the second colour, by default "RP"
    fig : :class:`~matplotlib.figure.Figure`, optional
        Figure on which to plot, by default will create a new one
    ax : :class:`~matplotlib.axes.Axes`, optional
        Axes on which to plot, by default will create a new one
    show : bool, optional
        Whether to immediately show the plot, by default True

    Returns
    -------
    fig, ax : :class:`~matplotlib.figure.Figure`
        Figure and axis that contains the plot
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 10))

        # flip the y-axis
        ax.invert_yaxis()

        # add a grid and axis labels
        ax.set_axisbelow(True)
        ax.grid(zorder=-1, linestyle="dotted", color="lightgrey")
        ax.set(xlabel=f'{c_filter_1} - {c_filter_2}', ylabel=m_filter)

        # create a custom colourmap based on the stellar types lists
        labels = [kstar_translator[i]["short"] for i in range(1, 13)]
        bounds = np.arange(1, 14)
        colours = [kstar_translator[i]["colour"] for i in range(1, 13)]
        cmap = mpl.colors.ListedColormap(colours)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                            spacing='proportional', label='Stellar type', ax=ax, pad=0.0)
        cbar.set_ticks(bounds[:-1] + 0.5, labels=labels, fontsize=0.5*fs)
        cbar.ax.tick_params(size=0.0)

    if "s" not in kwargs:
        kwargs["s"] = 10

    # plot everything up to neutron stars
    for kstar in range(13):
        # loop over bound binaries, disrupted primaries and disrupted secondaries
        for dis, marker, suffix in zip([False, True, True], ["o", "^", "v"], ["1", "1", "2"]):
            # select stars with the right kstar (based on which is brighter)
            mask = np.where(pop.observables["secondary_brighter"], 
                            pop.final_bpp["kstar_2"] == kstar,
                            pop.final_bpp["kstar_1"] == kstar)

            # apply disruption mask
            mask = (mask & pop.disrupted) if dis else (mask & ~pop.disrupted)

            ax.scatter(pop.observables[f'{c_filter_1}_app_{suffix}'][mask]
                       - pop.observables[f'{c_filter_2}_app_{suffix}'][mask],
                       pop.observables[f'{m_filter}_abs_{suffix}'][mask],
                       color=kstar_translator[kstar]["colour"],
                       marker=marker, **kwargs)

    if show:
        plt.show()

    return fig, ax


def _use_white_text(rgba):
    """Determines whether to use white text on a given background color based on the RGBA value.

    Parameters
    ----------
    rgba : `tuple`
        RGBA value

    Returns
    -------
    flag : `bool`
        True if white text should be used
    """
    r, g, b, _ = rgba
    return (r * 0.299 + g * 0.587 + b * 0.114) < 186 / 255


def _supernova_marker(ax, x, y, s):
    """Add a supernova marker to an ax

    Parameters
    ----------
    ax : :class:`~matplotlib.pyplot.axis`
        Axis on which to add marker
    x : `float`
        x coordinate
    y : `float`
        y coordinate
    s : `float`
        Size scale
    """
    ax.scatter(x, y, marker=(15, 1, 0), s=s * 6, zorder=-1,
               facecolor="#ebd510", edgecolor="#ebb810", linewidth=2)
    ax.scatter(x, y, marker=(10, 1, 0), s=s * 4, zorder=-1,
               facecolor="orange", edgecolor="#eb7e10", linewidth=2)


def _rlof_path(centre, width, height, m=1.5, flip=False):
    """Draw a path in the shape of RLOF (teardrop)

    Parameters
    ----------
    centre : `tuple`
        Centre of the teardrop (x, y)
    width : `float`
        How wide to make
    height : `float`
        How tall to make it
    m : `float`, optional
        The 'pointy-ness' of the teardrop, by default 1.5
    flip : `bool`, optional
        Whether to flip the direction, by default False

    Returns
    -------
    x, y : :class:`~numpy.ndarray`
        Path of the teardrop
    """
    t = np.linspace(0, 2 * np.pi, 1000)
    x = 0.5 * width * np.cos(t) * (-1 if flip else 1) + centre[0]
    y = height * np.sin(t) * np.sin(0.5 * t)**(m) + centre[1]

    return x, y


def plot_cartoon_evolution(bpp, bin_num, label_type="long", plot_title="Cartoon Binary Evolution",
                           y_sep_mult=1.5, offset=0.2, s_base=1000,
                           time_fs_mult=1.0, mass_fs_mult=1.0, kstar_fs_mult=1.0,
                           porb_fs_mult=1.0, label_fs_mult=1.0,
                           fig=None, ax=None, show=True):    # pragma: no cover
    """Plot COSMIC bpp output as a cartoon evolution

    Parameters
    ----------
    bpp : `pandas.DataFrame`
        COSMIC bpp table
    bin_num : `int`
        Binary number of the binary to plot
    label_type : `str`, optional
        What sort of annotated labels to use ["short", "long", "sentence"], by default "long"
    plot_title : `str`, optional
        Title to use for the plot, use "" for no title, by default "Cartoon Binary Evolution"
    y_sep_mult : `float`, optional
        Multiplier to use for the y separation (larger=more spread out steps, longer figure)
    offset : `float`, optional
        Offset from the centre for each of the stars (larger=wider binaries)
    s_base : `float`, optional
        Base scatter point size for the stars
    time_fs_mult : `float`, optional
        Multiplier for the time annotation fontsize
    mass_fs_mult : `float`, optional
        Multiplier for the mass annotation fontsize
    kstar_fs_mult : `float`, optional
        Multiplier for the kstar annotation fontsize
    porb_fs_mult : `float`, optional
        Multiplier for the porb annotation fontsize
    label_fs_mult : `float`, optional
        Multiplier for the evolution label fontsize
    fig : :class:`~matplotlib.pyplot.figure`, optional
        Figure on which to plot, by default will create a new one
    ax : :class:`~matplotlib.pyplot.axis`, optional
        Axis on which to plot, by default will create a new one
    show : `bool`, optional
        Whether to immediately show the plot, by default True

    Returns
    -------
    fig, ax : :class:`~matplotlib.pyplot.figure`, :class:`~matplotlib.pyplot.axis`
        Figure and axis of the plot
    """
    # extract the pertinent information from the bpp table
    df = bpp.loc[bin_num][["tphys", "mass_1", "mass_2", "kstar_1", "kstar_2", "porb", "sep",
                           "evol_type", "RRLO_1", "RRLO_2"]]

    # add some offset kstar columns to tell what type a star *previously* was
    df[["prev_kstar_1", "prev_kstar_2", "prev_evol_type"]] = df.shift(1, fill_value=0)[["kstar_1", "kstar_2",
                                                                                        "evol_type"]]

    # delete rows where RLOF ends immediately after a CE ends
    df = df[~((df["evol_type"] == 4) & (df["prev_evol_type"] == 8))]

    # count the number of evolution steps and start figure with size based on that
    total = len(df)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, total * y_sep_mult))

    # instantiate some flags to track state of binary
    i = 0
    disrupted = False
    common_envelope = False
    rlof = False
    contact = False

    min_log10_sep = np.log10(df["sep"][df["sep"] > 0.0].min())
    max_log10_sep = np.log10(df["sep"].max())

    # group timesteps and row indices by time
    times, row_inds = [], []
    prev_time, rows, j = -1.0, [], 0
    for _, row in df.iterrows():
        if row["tphys"].round(2) == prev_time:
            rows.append(j)
        else:
            if prev_time >= 0.0:
                times.append(prev_time)
                row_inds.append(rows)
            rows = [j]
            prev_time = row["tphys"].round(2)
        j += 1
    # append the last set of rows
    times.append(prev_time)
    row_inds.append(rows)

    # annotate the time on the left side of the binary
    for time, inds in zip(times, row_inds):
        ax.annotate(f'{time:1.2e} Myr' if time > 1e4 else f'{time:1.2f} Myr',
                    xy=(-offset - 0.3 - (0.12 if len(inds) > 1 else 0),
                        total - np.mean(inds) * y_sep_mult), ha="right", va="center",
                    fontsize=0.4*fs*time_fs_mult, fontweight="bold", zorder=-1,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="white") if len(inds) > 1 else None)
        # if there's more than one ind then plot a bracketed line connecting them
        if len(inds) > 1:
            ax.annotate('', xy=(-offset - 0.35, total - np.mean(inds) * y_sep_mult),
                        xytext=(-offset - 0.4, total - np.mean(inds) * y_sep_mult),
                        ha='center', va='center',
                        # 2.5 is a magic number here to make the bracket the right size
                        arrowprops=dict(arrowstyle=f'-[, widthB={y_sep_mult * len(inds) * 2.5}, lengthB=1',
                                        lw=1.5, color='k'))

    sep_offset = 0.2

    # go through each row of the evolution
    for _, row in df.iterrows():
        # use the translators to convert evol_type and kstars
        et_ind, k1, k2, pk1, pk2 = int(row["evol_type"]), kstar_translator[int(row["kstar_1"])],\
            kstar_translator[int(row["kstar_2"])], kstar_translator[int(row["prev_kstar_1"])],\
            kstar_translator[int(row["prev_kstar_2"])]
        et = evol_type_translator[et_ind]

        if row["sep"] > 0.0:
            sep_modifier = (np.log10(row["sep"]) - min_log10_sep) / (max_log10_sep - min_log10_sep)
            off_s = sep_offset * sep_modifier
        else:
            sep_modifier, off_s = None, 0.0

        # set disrupted, rlof and common-envelope flags are necessary
        if et_ind == 11 or row["porb"] < 0.0:
            disrupted = True
        if et_ind == 3 and (((row["RRLO_1"] >= 1.0) & (row["kstar_1"] < 13)) | (row["kstar_2"] < 13)):
            rlof = True
        if et_ind == 4:
            rlof = False
            contact = False
        if et_ind == 5:
            rlof = False
            contact = True
        if et_ind == 7:
            common_envelope = True
        if et_ind == 8:
            common_envelope = False
            rlof = False

        if row["RRLO_1"] < 1.0 and row["RRLO_2"] < 1.0:
            rlof = False
            common_envelope = False
            contact = False

        # check if either star is now a massless remnant
        mr_1 = k1["short"] == "MR"
        mr_2 = k2["short"] == "MR"
        ks_fontsize = 0.3 * fs * kstar_fs_mult

        # start an evolution label variable
        evol_label = et[label_type]

        # if the star just evolved then edit to label to explain what happened
        if et_ind == 2:
            which_star = "Primary" if k1 != pk1 else "Secondary"
            to_type = k1[label_type] if k1 != pk1 else k2[label_type]
            evol_label = f'{which_star} evolved to\n{to_type}'

        # annotate the evolution label on the right side of the binary
        ax.annotate(evol_label, xy=(0.5, total - i), va="center", fontsize=0.4*fs*label_fs_mult)

        # if we've got a common envelope then draw an ellipse behind the binary
        if common_envelope:
            envelope = mpl.patches.Ellipse(xy=(0, total - i),
                                           width=4 * offset + off_s, height=1.5 + off_s,
                                           facecolor="orange", edgecolor="none", zorder=-1, alpha=0.5)
            envelope_edge = mpl.patches.Ellipse(xy=(0, total - i),
                                                width=4 * offset + off_s, height=1.5 + off_s,
                                                facecolor="none", edgecolor="darkorange", lw=2)
            ax.add_artist(envelope)
            ax.add_artist(envelope_edge)

        # if either star is a massless remnant then we're just dealing with a single star now
        if mr_1 or mr_2:
            # plot the star centrally and a little larger
            ax.scatter(0, total - i, color=k1["colour"] if mr_2 else k2["colour"], s=s_base * 1.5)

            # label its stellar type if (a) it changed or (b) we're at the start/end of evolution
            if (k1 != pk1 and not mr_1) or (k2 != pk2 and not mr_2) or et_ind in [1, 10]:
                ax.annotate(k1["short"] if mr_2 else k2["short"], xy=(0, total - i),
                            ha="center", va="center", zorder=10, fontsize=ks_fontsize, fontweight="bold",
                            color="white" if _use_white_text(k1["colour"]
                                                             if mr_2 else k2["colour"]) else "black")

            # annotate the correct mass
            ax.annotate(f'{row["mass_1"] if mr_2 else row["mass_2"]:1.2f} ' + r'$\rm M_{\odot}$',
                        xy=(0, total - i - 0.45), ha="center", va="top", fontsize=0.3*fs*mass_fs_mult,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                        if et_ind in [15, 16] else None)

            # if a supernova just happened then add an explosion marker behind the star
            if et_ind in [15, 16]:
                _supernova_marker(ax, 0, total - i, s_base)

        # otherwise we've got two stars
        else:
            contact_adjust = 0.25 if contact else 1.0

            # plot stars offset from the centre
            ax.scatter(0 - (offset + off_s) * contact_adjust, total - i,
                       color=k1["colour"], s=s_base, zorder=10)
            ax.scatter(0 + (offset + off_s) * contact_adjust, total - i,
                       color=k2["colour"], s=s_base, zorder=10)

            # annotate the mass (with some extra padding if there's RLOF)
            mass_y_offset = 0.35 if not (rlof and not common_envelope) else 0.5
            ax.annotate(f'{row["mass_1"]:1.2f} ' + r'$\rm M_{\odot}$',
                        xy=(0 - offset * contact_adjust - off_s, total - i - mass_y_offset),
                        ha="left" if common_envelope else "center", va="top", fontsize=0.3*fs*mass_fs_mult,
                        rotation=45 if contact else 0,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                        if et_ind in [15, 16] else None)
            ax.annotate(f'{row["mass_2"]:1.2f} ' + r'$\rm M_{\odot}$',
                        xy=(0 + offset * contact_adjust + off_s, total - i - mass_y_offset),
                        ha="right" if common_envelope else "center", va="top", fontsize=0.3*fs*mass_fs_mult,
                        zorder=1000,
                        rotation=45 if contact else 0,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                        if et_ind in [15, 16] else None)

            # if the primary type changed or we're at the start/end then label it
            if k1 != pk1 or et_ind in [1, 10]:
                ax.annotate(k1["short"], xy=(0 - offset * contact_adjust - off_s, total - i),
                            ha="center", va="center",
                            color="white" if _use_white_text(k1["colour"]) else "black",
                            zorder=10, fontsize=ks_fontsize, fontweight="bold")

            # if the secondary type changed or we're at the start/end then label it
            if k2 != pk2 or et_ind in [1, 10]:
                ax.annotate(k2["short"], xy=(0 + offset * contact_adjust + off_s, total - i),
                            ha="center", va="center",
                            color="white" if _use_white_text(k2["colour"]) else "black",
                            zorder=10, fontsize=ks_fontsize, fontweight="bold")

            # for bound binaries plot a line connecting them
            if not disrupted or et_ind == 11:
                ax.plot([0 - offset * contact_adjust - off_s, 0 + offset * contact_adjust + off_s], [total - i, total - i],
                        linestyle="--", zorder=-1, color="black")

            if et_ind == 11:
                ax.scatter(0, total - i, marker=(10, 1, 360 / 10 / 2), s=s_base / 2, zorder=-1,
                           facecolor="orange", edgecolor="none", linewidth=1)

            if not disrupted:
                # annotate the line with period, offset to one side if there's RLOF
                x = 0 if not (rlof and not common_envelope) else (-offset * contact_adjust / 4 if row["RRLO_1"] >= 1.0 else offset * contact_adjust / 4)
                p_lab = f'{row["porb"]:1.2e} days' if row["porb"] > 10000 or row["porb"] < 1\
                    else f'{row["porb"]:1.0f} days'
                ax.annotate(p_lab, xy=(x, total - i + 0.05), ha="center", va="bottom",
                            fontsize=0.2*fs*porb_fs_mult if row["porb"] > 10000 or row["porb"] < 1 else 0.3*fs*porb_fs_mult)

            # for non-common-envelope RLOF, plot a RLOF teardrop in the background
            if rlof and not common_envelope:
                # flip the shape depending on the direction
                if row["RRLO_1"] >= 1.0:
                    x, y = _rlof_path((0 - offset / 2.6, total - i), 2 * (offset + off_s),
                                      0.6 * (1 + off_s), flip=False)
                else:
                    x, y = _rlof_path((0 + offset / 2.6, total - i), 2 * (offset + off_s),
                                      0.6 * (1 + off_s), flip=True)
                ax.plot(x, y, color="darkorange", lw=2)
                ax.fill_between(x, y, color="orange", alpha=0.5, edgecolor="none", zorder=-2)

            # add supernova explosion markers as necessary
            if et_ind == 15:
                _supernova_marker(ax, 0 - offset * contact_adjust - off_s, total - i, s_base / 1.5)
            if et_ind == 16:
                _supernova_marker(ax, 0 + offset * contact_adjust + off_s, total - i, s_base / 1.5)

        # increment by multiplier
        i += y_sep_mult

    # clear off any x-ticks and axes
    ax.set_xlim(-1.5, 1.5)
    ax.set_xticks([])
    ax.axis("off")

    # annotate a title as the top
    ax.annotate(plot_title, xy=(0, total + 0.75), ha="center", va="center", fontsize=fs * 1.2)

    if show:
        plt.show()

    return fig, ax


def plot_galactic_orbit(primary_orbit, secondary_orbit=None,
                        t_min=0 * u.Myr, t_max=np.inf * u.Myr, show_start=True,
                        primary_kwargs={}, secondary_kwargs={}, start_kwargs={},
                        show_legend=True,
                        fig=None, axes=None, show=True):
    """Plot the galactic orbit of a binary system. This provides a wrapper around the gala
    :class:`~gala.dynamics.Orbit` method :meth:`~gala.dynamics.Orbit.plot`.

    Parameters
    ----------
    primary_orbit : :class:`~gala.dynamics.Orbit`
        Orbit of the primary star (or bound binary)
    secondary_orbit : :class:`~gala.dynamics.Orbit`, optional
        Orbit of the secondary star, by default None (no disruption)
    t_min : :class:`~astropy.units.Quantity` [time], optional
        Minimum time since the start of the orbits to plot, by default 0*u.Myr
    t_max : :class:`~astropy.units.Quantity` [time], optional
        Maximum time since the start of the orbits to plot, by default np.inf*u
    show_start : `bool`, optional
        Whether to plot a marker at the start of the orbits, by default True
    primary_kwargs : `dict`, optional
        Keyword arguments to pass to the primary orbit plot, by default {}
    secondary_kwargs : `dict`, optional
        Keyword arguments to pass to the secondary orbit plot, by default {}
    start_kwargs : `dict`, optional
        Keyword arguments to pass to the start marker plot, by default {}
    fig : :class:`~matplotlib.figure.Figure`, optional
        Figure on which to plot, by default will create a new one
    axes : :class:`~matplotlib.axes.Axes`, optional
        Axes on which to plot, by default will create a new one
    show : `bool`, optional
        Whether to immediately show the plot, by default

    Returns
    -------
    fig, axes : :class:`~matplotlib.figure.Figure`, :class:`~matplotlib.axes.Axes`
        Figure and axes of the plot
    """
    # create a mask for the times
    time_mask = ((primary_orbit.t - primary_orbit.t[0] >= t_min)
                 & (primary_orbit.t - primary_orbit.t[0] < t_max))

    if not time_mask.any():
        raise ValueError("No times in the specified range. Please check t_min and t_max.")

    # add a label to the primary orbit if not already present
    if "label" not in primary_kwargs:
        primary_kwargs["label"] = "Primary orbit"

    # plot the primary orbit and save the resulting figure
    fig = primary_orbit[time_mask].plot(axes=axes, **primary_kwargs)

    # same thing for secondary orbit (if it exists)
    if secondary_orbit is not None:
        if "label" not in secondary_kwargs:
            secondary_kwargs["label"] = "Secondary orbit"
        # find the first time at which the secondary orbit differs from the primary orbit
        t_diffs = np.where(np.any(secondary_orbit[time_mask].xyz != primary_orbit[time_mask].xyz, axis=0))[0]

        if len(t_diffs) != 0:
            t_diff = max(0, t_diffs[0] - 1)

            # only plot the secondary orbit if it differs from the primary orbit
            secondary_orbit[time_mask][t_diff:].plot(axes=fig.axes, **secondary_kwargs)

    # if we're showing the start position then plot a marker there
    if show_start:
        # use some default settings that the user can update
        full_start_kwargs = {
            "marker": 'o',
            "color": 'black',
            "s": 50,
            "label": "Start position"
        }
        full_start_kwargs.update(start_kwargs)

        primary_orbit[0].plot(axes=fig.axes, **full_start_kwargs)

    if show_legend:     # pragma: no cover
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=16)
        fig.subplots_adjust(top=0.9)

    # show the plot if desired
    if show:
        plt.show()

    return fig, fig.axes
