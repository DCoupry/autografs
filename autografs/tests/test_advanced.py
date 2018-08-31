#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .context import autografs

import unittest

logger = logging.getLogger(__name__) 


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_autografs(self):
    	logger.debug("Testing full Autografs functionalities.")
    	mofgen = autografs.Autografs()
		topology_name = "hcb"
		sbu_names = [("Benzene_linear",0.5),("Acetylene_linear",0.5),"Triphenylene_boronated_triangle"]
		mof = mofgen.make(topology_name=topology_name,sbu_names=sbu_names,coercion=True, supercell=(2,2,2))
		del mof[0]
		sites = mof.list_functionalizable_sites(sbu_names=["Benzene_linear"])
		for site in sites:
			mof.functionalize(where=site,fg=mofgen.sbu["Amine_cap"])
		# flip, deletion, defect, proba, supercells
		# compare
        return True


if __name__ == '__main__':
	unittest.main()