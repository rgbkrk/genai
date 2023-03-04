# from IPython.testing.globalipapp import get_ipython

# ip = get_ipython()

from genai import __version__

import unittest


class TestSuggestions(unittest.TestCase):
    def test_version(self):
        assert __version__ == "0.9.0"
