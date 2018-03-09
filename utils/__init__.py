#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__  = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = 2.0
__status__  = "alpha"

import os

__all__     = ["pointgroup","topologies","sbu","operations","mmanalysis","io"]
__data__    = os.path.join("/".join(os.path.dirname(__file__).split("/")[:-1]),"data")
