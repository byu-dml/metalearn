import unittest

from metalearn.metafeatures.base import collectordict


class CollectorDictTestCase(unittest.TestCase):

    def test_no_init_args(self):
        try:
            cd = collectordict({'a': 1})
            self.fail('collectordict should have failed when passed init args')
        except TypeError as e:
            pass

    def test_no_duplicate_setter(self):
        cd = collectordict()
        cd[1] = 1
        try:
            cd[1] = 2
            self.fail('collectordict should have raised an error when setting an existing key')
        except LookupError as e:
            pass

    def test_no_duplicates_in_update(self):
        cd = collectordict()
        cd[1] = 1
        try:
            cd.update({1:2})
            self.fail('collectordict should have raised an error when updating with an existing key')
        except LookupError as e:
            pass
