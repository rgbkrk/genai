__version__ = "0.1.0"

"""
Genai IPython magic extensions.

Magic methods:

%%assist 

Usage:
  %load_ext genai
"""

from .assist import assist


def load_ipython_extension(ipython):
    pass
