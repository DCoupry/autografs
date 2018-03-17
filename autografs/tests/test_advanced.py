#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from .context import autografs

import unittest

logger = logging.getLogger(__name__) 


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_autografs(self):
    	logger.debug("Testing Autografs.")
        return True


if __name__ == '__main__':
	unittest.main()