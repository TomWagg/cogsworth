"""Checks for optional dependencies using lazy import from
`PEP 562 <https://www.python.org/dev/peps/pep-0562/>`_.

I brazenly stole this from Gala (see `here
<https://github.com/adrn/gala/blob/3d761a9fd1447fbcf0f319c6fe87a0d4f6a5ceed/gala/tests/optional_deps.py>`_)
and made some changes to the error message
"""
import importlib
from collections.abc import Sequence
import logging

# First, the top-level packages:
# TODO: This list is a duplicate of the dependencies in setup.cfg "all", but
# some of the package names are different from the pip-install name (e.g.,
# beautifulsoup4 -> bs4).
_optional_deps = ['nose', 'tables', 'isochrones', 'dustmaps', 'healpy', 'gaiaunlimited', 'agama',
                  'legwork', 'pynbody']
_purposes = ['observables predictions', 'observables predictions', 'observables predictions',
             'observables predictions', 'healpix maps', 'GAIA observation predictions',
             'action-based potentials', 'LISA gravitational wave predictions',
             'loading hydrodynamical snapshots']
_deps = {k: (k, p) for k, p in zip(_optional_deps, _purposes)}

# Any subpackages that have different import behaviour:
_deps['matplotlib'] = (['matplotlib', 'matplotlib.pyplot'], 'visualisation')

# __all__ = [f"HAS_{pkg}" for pkg in _deps]
__all__ = ["check_dependencies"]


def check_dependencies(names):
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in _deps.keys():
            module_name = name
            modules, purpose = _deps[module_name]

            if not isinstance(modules, Sequence) or isinstance(modules, str):
                modules = [modules]

            for module in modules:
                try:
                    if module_name == "isochrones":
                        # HACK around the isochrones import to ignore warnings about Holoview and Multinest
                        logging.getLogger("isochrones").setLevel("ERROR")
                    importlib.import_module(module)
                    if module_name == "isochrones":
                        logging.getLogger("isochrones").setLevel("WARNING")

                except (ImportError, ModuleNotFoundError):
                    raise ImportError((f"`{module_name}` required for {purpose} with cogsworth\n"
                                       "Either install this package directly, or install all optional "
                                       "`cogsworth` dependencies with `pip install cogsworth[all]`\n"))
        else:
            raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
    return True
