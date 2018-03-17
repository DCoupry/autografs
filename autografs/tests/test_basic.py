#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from .context import autografs

import unittest

logger = logging.getLogger(__name__) 


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_symmetry_detection(self)

		logger.debug("Testing symmetry detection.")
		cwd = os.path.dirname(__file__)
		assertions = []
		for T in ["C3h","Ci","D3d","D3h","D5d","Oh","Td"]:
			molpath = os.path.join(cwd,"test/{0}.xyz".format(T))
			logpath = os.path.join(cwd,"test/{0}.log".format(T))
			mol = read(molpath)
			pg = PointGroup(mol,tol=0.3)
			F = pg.schoenflies
			R = (T==F)
			logger.debug("Test {T} is {R} --> {F}".format(T=T,F=F,R=R))
			assertions.append(R)
		return all(R)

	# TODO 
	def test_rotation(self):
		logger.debug("Testing rotation operation.")
		return True

	def test_reflection(self):
		logger.debug("Testing reflection operation.")
		return True

	def test_mmanalysis(self):
		logger.debug("Testing bonding and FF parametrization analysis.")
		return True



if __name__ == '__main__':
	unittest.main()
