"""Genai: generative AI tooling for IPython and Jupyter notebooks

## Magic methods

    - `%%assist`: Assist you in writing code.

## Usage

  `%load_ext genai` to load the extension in other notebook
  projects
"""


def load_ipython_extension(ipython):
    import genai.suggestions
    from genai.magics import assist, prompt

    ipython.register_magic_function(assist, "cell")
    ipython.register_magic_function(prompt, "cell")

    genai.suggestions.register()


def unload_ipython_extension(ipython):
    # Unload the custom exception handler
    ipython.set_custom_exc((Exception,), None)
