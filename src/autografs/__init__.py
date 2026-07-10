#!/usr/bin/env python3
# Copyright : see accompanying license files for details
"""
AuToGraFS: Automatic Topological Generator for Framework Structures.

This package provides tools for generating Metal-Organic Frameworks (MOFs)
and other periodic crystalline structures from topological blueprints.

Modules
-------
builder
    Main framework builder class for generating structures.
fragment
    Fragment data structure for molecular fragments.
topology
    Topology data structure for periodic blueprints.
utils
    Utility functions for file I/O, graph manipulation, and visualization.

Classes
-------
Autografs
    Main class for building framework structures from topologies and SBUs.
Fragment
    Represents a molecular fragment with symmetry information.
Topology
    Represents a periodic topology blueprint.

Examples
--------
Basic usage:

>>> from autografs import Autografs
>>> mofgen = Autografs()
>>> topologies = mofgen.list_topologies()
>>> sbu_dict = mofgen.list_building_units(sieve="pcu")

References
----------
.. [1] Addicoat, M., Coupry, D. E., & Heine, T. (2014).
       AuToGraFS: Automatic Topological Generator for Framework Structures.
       The Journal of Physical Chemistry A, 118(40), 9607-14.
"""

from __future__ import annotations

__author__ = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = "3.0.0"
__status__ = "production"
__all__ = [
    "utils",
    "alignment",
    "fragment",
    "framework",
    "framework_io",
    "net",
    "porosity",
    "topology",
    "topology_io",
    "builder",
    "exceptions",
    "relax",
    "Autografs",
    "AlignmentError",
    "AutografsError",
    "Fragment",
    "Framework",
    "NetMismatchError",
    "OverlapError",
    "RelaxationError",
    "Topology",
]

import logging

# bind every module listed in __all__ as a package attribute; relax's
# optional LAMMPS backends are imported lazily inside its functions,
# so importing the module itself is cheap and always safe
from autografs import framework_io, net, porosity, relax
from autografs.builder import Autografs
from autografs.exceptions import (
    AlignmentError,
    AutografsError,
    NetMismatchError,
    OverlapError,
    RelaxationError,
)
from autografs.fragment import Fragment
from autografs.framework import Framework
from autografs.topology import Topology

# Add NullHandler to prevent "No handler found" warnings
# Applications should configure their own logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
