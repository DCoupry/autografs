#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
__all__ = ["utils", "fragment", "topology", "builder", "Autografs", "Fragment", "Topology"]

import logging

from autografs.builder import Autografs
from autografs.fragment import Fragment
from autografs.topology import Topology

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(
    format="[AuToGraFS] %(asctime)s | %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)
