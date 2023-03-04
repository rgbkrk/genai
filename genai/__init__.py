"""Genai: generative AI tooling for IPython and Jupyter notebooks

## Magic methods

    - `%%assist`: Assist you in writing code.

## Usage

  `%load_ext genai` to load the extension in other notebook
  projects
"""

__version__ = "0.8.0"


def load_ipython_extension(ipython):
    from .suggestions import register
    from .assist import assist

    register()
