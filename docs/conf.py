"""Sphinx configuration for the SAF-W documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SAF-W"
author = "Tianshuai Gao"
release = "1.0.0"
version = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
}

master_doc = "index"
html_theme = "sphinx_rtd_theme"
templates_path = ["_templates"]
exclude_patterns = ["_build"]
