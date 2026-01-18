# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------
project = 'dmipy-jax'
copyright = '2026, Rutger Fick, Demian Wassermann, and Contributors'
author = 'Rutger Fick, Demian Wassermann'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'myst_parser',          # Markdown support
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['references.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = "Dmipy-JAX Documentation"

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('..'))

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "tasklist",
]
