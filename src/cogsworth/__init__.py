from . import kicks, pop, events, classify, obs, plot, sfh, utils, hydro, interop
from ._version import __version__
from .citations import CITATIONS

from .pop import Population
from .interop.compas.pop import COMPASPopulation

__bibtex__ = __citation__ = CITATIONS["general"]["cogsworth"]["bibtex"]
__uri__ = "https://cogsworth.readthedocs.io/"
__author__ = "Tom Wagg"
__email__ = "tomjwagg@gmail.com"
