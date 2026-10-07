# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sphinx configuration for the sbd-eigensolver documentation."""

import inspect
import os
import re
import sys
from importlib.metadata import version as metadata_version

# Note there is deliberately no `sys.path` manipulation here. The import name of this
# package (`sbd`) does not match the directory it lives in (`python/`); the mapping is
# declared as `package-dir = {sbd = "python"}` in pyproject.toml and only setuptools
# knows about it. Autodoc therefore needs the package genuinely *installed*, which is
# also what makes the `metadata_version` call below work.

project = "Selected Basis Diagonalization (SBD)"
project_copyright = "2026, IBM Quantum"
description = "Python bindings for the SBD eigensolver library"
author = "IBM Quantum"
language = "en"
# The *distribution* name ("sbd-eigensolver"), not the import name ("sbd").
release = metadata_version("sbd-eigensolver")

html_theme = "qiskit-ecosystem"

html_theme_options = {
    "dark_logo": "images/qiskit-dark-logo.svg",
    "light_logo": "images/qiskit-light-logo.svg",
    "sidebar_qiskit_ecosystem_member": False,
}
html_static_path = ["_static"]
templates_path = ["_templates"]

# Sphinx should ignore these patterns when building.
exclude_patterns = [
    "_build",
]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "reno.sphinxext",
    "qiskit_sphinx_theme",
]

html_last_updated_fmt = "%Y/%m/%d"
html_title = f"{project} {release}"

# This allows RST files to put `|version|` in their file and
# have it updated with the release set in conf.py.
rst_prolog = f"""
.. |version| replace:: {release}
"""

# Options for autodoc. These reflect the values from Qiskit SDK and Runtime.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# This adds numbers to the captions for figures, tables,
# and code blocks.
numfig = True
numfig_format = {"table": "Table %s"}

add_module_names = False

modindex_common_prefix = ["sbd."]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------

# The package is imported as `sbd` but stored in the repository under `python/`, so the
# two halves of a source link need different spellings: the installed file path is
# split on one, and the URL is built with the other.
_IMPORT_NAME = "sbd"
_REPO_SUBDIR = "python"


def determine_github_branch() -> str:
    """Determine the GitHub branch name to use for source code links.

    We need to decide whether to use `stable/<version>` vs. `main` for dev builds.
    Refer to https://docs.github.com/en/actions/learn-github-actions/variables
    for how we determine this with GitHub Actions.
    """
    # If CI env vars not set, default to `main`. This is relevant for local builds.
    if "GITHUB_REF_NAME" not in os.environ:
        return "main"

    # PR workflows set the branch they're merging into.
    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        return base_ref

    ref_name = os.environ["GITHUB_REF_NAME"]

    # Check if the ref_name is a tag like `1.0.0` or `1.0.0rc1`. If so, we need
    # to transform it to a Git branch like `stable/1.0`.
    version_without_patch = re.match(r"(\d+\.\d+)", ref_name)
    return f"stable/{version_without_patch.group()}" if version_without_patch else ref_name


GITHUB_BRANCH = determine_github_branch()


def linkcode_resolve(domain, info):
    """Point the "source" link of each documented object at GitHub."""
    if domain != "py":
        return None

    module_name = info["module"]
    module = sys.modules.get(module_name)
    if module is None or _IMPORT_NAME not in module_name:
        return None

    def is_valid_code_object(obj):
        return inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj)

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
        if not is_valid_code_object(obj):
            return None

    # Unwrap decorators. This requires they used `functools.wrap()`.
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
        if not is_valid_code_object(obj):
            return None

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None or f"/{_IMPORT_NAME}/" not in full_file_name:
        return None
    file_name = full_file_name.split(f"/{_IMPORT_NAME}/")[-1]

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"
    return (
        "https://github.com/Qiskit/sbd-eigensolver-python/tree/"
        f"{GITHUB_BRANCH}/{_REPO_SUBDIR}/{file_name}{linespec}"
    )
