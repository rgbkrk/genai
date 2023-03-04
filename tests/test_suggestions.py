import unittest
from unittest.mock import MagicMock

from genai import suggestions


class TestSuggestions(unittest.TestCase):
    def test_register(self):
        fake_ip = MagicMock()
        suggestions.register(fake_ip)

        fake_ip.set_custom_exc.assert_called_once()
