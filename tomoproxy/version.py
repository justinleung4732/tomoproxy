#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools to report version information for LEMA

This can also report modle versions of prereq
packages.
"""
import sys
import os
import inspect
import warnings
import numpy as np
import scipy as sp
import healpy as hp
import burnman
import pyshtools as shtools
try:
    import git
    have_git = True
except ImportError:
    have_git = False
try:
    import numba
    have_numba = True
except ImportError:
    have_numba = False

# Our version number. Bump this for each release
# (which should also correspond to a git tag) and
# use semantic versioning.
__version__ = "0.0.1"

def collect_version():
    """
    Return a dictionary of version information

    Dictionary keys are:
    * "git_repo" - True if a git repo, False if not, None if git module not
                   installed
    * "git_sha" - hex sha git commit (if git module is installed and 
                  git_repo is True) or None
    * "git_commit" - 'title' of the head git commit
    * "numba" - numba version (or None if not installed)
    * "numpy" - numpy version
    * "scipy" - scipy version
    * "healpy" - healpy version
    * "shtools" - shtools version
    * "burnman" - burnman version
    * "python" - python version info
    """
    version_dict = {}
    if have_numba:
        version_dict["numba"] = numba.__version__ 
    else:
        version_dict["numba"] = None
    version_dict["numpy"] = np.__version__
    version_dict["scipy"] = sp.__version__
    version_dict["healpy"] = hp.__version__
    version_dict["shtools"] = shtools.__version__
    version_dict["burnman"] = burnman.__version__
    version_dict["python"] = sys.version
    if have_git:
        root_dir = os.path.join(os.path.dirname(os.path.abspath(
          inspect.getfile(inspect.currentframe()))), "..")
        try:
            git_repo = git.Repo(root_dir)
            version_dict["git_repo"] = True
            version_dict["git_sha"] = git_repo.head.commit.hexsha
            version_dict["git_commit"] = \
                git_repo.head.commit.message.split('\n')[0]
        except git.exc.InvalidGitRepositoryError:
            # Probably not a git repo
            version_dict["git_repo"] = False
            version_dict["git_sha"] = None
            version_dict["git_commit"] = None
    else:
        version_dict["git_repo"] = None
        version_dict["git_sha"] = None
        version_dict["git_commit"] = None
    return version_dict


def report_version(numba_warn=True):
    """
    Print the version info

    If numba is not installed and numba_warn is True, tell the
    user to install numba
    """
    version_dict = collect_version()
    print("LEMA version", __version__)
    if version_dict["git_repo"]:
        print("Running from git repository with head at:")
        print("    ", version_dict["git_commit"]) 
        print("     sha:", version_dict["git_sha"])
    print("Using:")
    print("  numba", version_dict["numba"])
    print("  numpy", version_dict["numpy"])
    print("  scipy", version_dict["scipy"])
    print("  healpy", version_dict["healpy"])
    print("  burnman", version_dict["burnman"])
    print("Python version is:\n", version_dict["python"])

    if numba_warn and (version_dict["numba"] is None):
        warnings.warn("Numba is not installed.") 
        print("Warning: Numba is not installed.\n",
              "Note that installing the Numba package should result in",
              "much shorter runtimes")
