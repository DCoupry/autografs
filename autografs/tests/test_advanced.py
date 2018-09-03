#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .context import autografs

import unittest

logger = logging.getLogger(__name__) 


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_autografs(self):
    	logger.debug("Testing full Autografs functionalities.")
		import random
		mofgen = autografs.Autografs()
		topology_name = "hcb"
		sbu_names = [("Benzene_linear",0.5),("Acetylene_linear",0.5),"Triphenylene_boronated_triangle"]
		mof = mofgen.make(topology_name=topology_name,sbu_names=sbu_names,coercion=True, supercell=(2,2,1))
		del mof[9]
		sites = mof.list_functionalizable_sites(sbu_names=["Benzene_linear"])
		caps = ["Amine_cap","Methyl_cap","Fluorine_cap"]
		for site in sites:
			if random.uniform(0,1)<0.25:
				cap = random.choice(caps)
				mof.functionalize(where=site,fg=mofgen.sbu[cap])
		supercell = mof.get_supercell(m=(2,2,1))
		for idx, sbu in supercell:
			try:
				supercell.rotate(idx,45.0)
			except Exception:
				continue
		# compare
        return True


if __name__ == '__main__':
	unittest.main()