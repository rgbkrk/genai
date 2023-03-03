"""Genai IPython magic extensions.

## Magic methods

    - `%%assist`: Assist you in writing code.

## Usage

  `%load_ext genai` to load the extension in other notebook
  projects
"""

__version__ = "0.8.0"

from .assist import assist


def load_ipython_extension(ipython):
    pass
