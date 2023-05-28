# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import inspect
import pathlib
import datetime
sys.path.insert(0, os.path.abspath('.'))

# HACKS - credit to "https://github.com/rodluger/starry_process"
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import hacks

from configparser import ConfigParser
conf = ConfigParser()

docs_root = pathlib.Path(__file__).parent.resolve()
conf.read([str(docs_root / '..' / 'setup.cfg')])
setup_cfg = dict(conf.items('metadata'))

# -- Project information -----------------------------------------------------

project = setup_cfg['name']
author = setup_cfg['author']
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, setup_cfg['author'])


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    'sphinx_design',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'numpydoc',
    'sphinxcontrib.bibtex',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinx.ext.linkcode',
    'sphinx_gallery.gen_gallery'
]

sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'reference_url': {
        'sphinx_gallery': None,
    },
    'download_all_examples': False,
    'show_signature': False,
    'matplotlib_animations': True
}


intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'matplotlib': ('https://matplotlib.org/stable', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy', None),
                       'astropy': ('https://docs.astropy.org/en/stable', None),
                       'pandas': ('https://pandas.pydata.org/docs/', None),
                       'gala': ('http://gala.adrian.pw/en/latest/', None),
                       'cosmic': ('https://cosmic-popsynth.github.io/docs/stable/', None)}

bibtex_bibfiles = ['tutorials/refs.bib']

# fix numpydoc autosummary
numpydoc_show_class_members = False

# use blockquotes (numpydoc>=0.8 only)
numpydoc_use_blockquotes = True

# auto-insert plot directive in examples
numpydoc_use_plots = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "logo": {
        "link": "index",
        "image_light": "gala_invite_light.png",
        "image_dark": "gala_invite_dark.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TomWagg/cosmic-gala",
            "icon": "fab fa-github-square",
        },
    ],
    "footer_start": ["copyright", "last-updated"],
    "secondary_sidebar_items": [],
    "header_links_before_dropdown": 7
}

html_last_updated_fmt = "%Y %b %d at %H:%M:%S UTC"
html_show_sourcelink = False
html_favicon = "_static/favicon.ico"

html_sidebars = {
    "index": [],
    "pages/install": [],
    "pages/getting_started": ["page-toc"],
    "tutorials/*": ["page-toc"],
    "**": ["sidebar-nav-bs.html"]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["signika.css", "custom.css"]
html_js_files = ['custom.js']

# autodocs
autoclass_content = "both"
autosummary_generate = True
autodoc_docstring_signature = True

# todos
todo_include_todos = True

# nbsphinx
nbsphinx_epilog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. note:: This tutorial was generated from a Jupyter notebook that can be
          `found here <https://github.com/TomWagg/cosmic-gala/tree/main/docs/{{ docname }}>`_.
"""
nbsphinx_prompt_width = "0"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

mathjax3_config = {
    'tex': {'tags': 'ams', 'useLabelIds': True},
}

def linkcode_resolve(domain, info):
    """function for linkcode sphinx extension"""
    def find_func():
        # find the installed module in sys module
        sys_mod = sys.modules[info["module"]]

        # use inspect to find the source code and starting line number
        names = info["fullname"].split(".")
        func = sys_mod
        for name in names:
            func = getattr(func, name)
        source_code, line_num = inspect.getsourcelines(func)

        # get the file name from the module
        file = info["module"].split(".")[-1]

        return file, line_num, line_num + len(source_code) - 1

    # ensure it has the proper domain and has a module
    if domain != 'py' or not info['module']:
        return None

    # attempt to cleverly locate the function in the file
    try:
        file, start, end = find_func()
        # stitch together a github link with specific lines
        filename = "cogsworth/{}.py#L{}-L{}".format(file, start, end)

    # if you can't find it in the file then just link to the correct file
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    return "https://github.com/TomWagg/cosmic-gala/blob/main/{}".format(filename)
