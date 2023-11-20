import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .utils import kstar_translator

__all__ = ["plot_cmd"]


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


def plot_cmd(pop, m_filter="G", c_filter_1="BP", c_filter_2="RP",
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
            # select primary stars with the right kstar
            mask = pop.final_bpp["kstar_1"] == kstar
            # use secondaries in cases where they are brighter
            mask[pop.observables["secondary_brighter"]] = (pop.final_bpp["kstar_2"] == kstar)

            # apply disruption mask
            mask = mask & pop.disrupted if dis else mask & ~pop.disrupted

            ax.scatter(pop.observables[f'{c_filter_1}_app_{suffix}'][mask]
                       - pop.observables[f'{c_filter_2}_app_{suffix}'][mask],
                       pop.observables[f'{m_filter}_abs_{suffix}'][mask],
                       color=kstar_translator[kstar]["colour"],
                       marker=marker, **kwargs)

    if show:
        plt.show()

    return fig, ax
