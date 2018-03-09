# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

"""
AuToGraFS: Automatic Topological Generator for Framework Structures.
Addicoat, M. a, Coupry, D. E., & Heine, T. (2014).
The Journal of Physical Chemistry. A, 118(40), 9607–14. 
"""

import math
import os

__all__     = ["autografs","framework","utils"]
__version__ = 2.0

from autografs.autografs import Autografs
from autografs.framework import Framework
