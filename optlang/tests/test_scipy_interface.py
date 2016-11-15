# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import unittest


try:
    import scipy
except ImportError as e:
    if str(e).find('scipy') >= 0:
        class TestMissingDependency(unittest.TestCase):

            @unittest.skip('Missing dependency - ' + str(e))
            def test_fail(self):
                pass
    else:
        raise
else:
    from optlang import scipy_interface

    class ScipyInterfaceTestCase(unittest.TestCase):
        pass
