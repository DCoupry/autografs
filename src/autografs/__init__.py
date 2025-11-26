#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright : see accompanying license files for details

__author__ = "Damien Coupry"
__credits__ = ["Prof. Matthew Addicoat"]
__license__ = "MIT"
__maintainer__ = "Damien Coupry"
__version__ = '3.0.0'
__status__ = "production"
__all__ = ["utils", "structure", "builder"]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(format='[AuToGraFS] %(asctime)s | %(message)s',
                    level=logging.INFO,
                    datefmt='%I:%M:%S')
