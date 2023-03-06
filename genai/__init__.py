"""Genai: generative AI tooling for IPython and Jupyter notebooks

## Magic methods

    - `%%assist`: Assist you in writing code.

## Usage

  `%load_ext genai` to load the extension in other notebook
  projects
"""

__version__ = "0.11.0"


def load_ipython_extension(ipython):
    import genai.suggestions
    from genai.magics import assist

    ipython.register_magic_function(assist, "cell")

    genai.suggestions.register()


def unload_ipython_extension(ipython):
    # Unload the custom exception handler
    ipython.set_custom_exc((Exception,), None)
