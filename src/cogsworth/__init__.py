from ._version import __version__

__uri__ = "https://cogsworth.readthedocs.io/"
__author__ = "Tom Wagg"
__email__ = "tomjwagg@gmail.com"

_lazy = {
    "kicks", "pop", "events", "classify", "obs", "plot", "sfh", "utils", "hydro", "interop",
    "Population", "COMPASPopulation", "CITATIONS", "__bibtex__", "__citation__",
}


def __getattr__(name):
    import importlib
    if name in ("kicks", "pop", "events", "classify", "obs", "plot", "sfh", "utils", "hydro", "interop"):
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name == "Population":
        from .pop import Population
        globals()["Population"] = Population
        return Population
    if name == "COMPASPopulation":
        from .interop.compas.pop import COMPASPopulation
        globals()["COMPASPopulation"] = COMPASPopulation
        return COMPASPopulation
    raise AttributeError(f"module 'cogsworth' has no attribute {name!r}")
