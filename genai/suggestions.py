"""
This module creates a custom exception handler that will send the error to OpenAI's
ChatGPT in order to help debug the code. It will also display the error in the
notebook as usual.
"""


from IPython.core.ultratb import AutoFormattedTB
from IPython.core.display import Markdown, display
from IPython import get_ipython

import openai

# initialize the formatter for making the tracebacks into strings
itb = AutoFormattedTB(mode="Plain", tb_offset=1)

# faking filename injection
notebook_name = "Exploring Python.ipynb"

# this function will be called on exceptions in any cell
def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    # Get the current code
    code = shell.user_ns["In"][-1]

    # still show the error within the notebook, don't just swallow it
    # shell.showtraceback((etype, evalue, []), tb_offset=tb_offset)
    display(Markdown(f"**{etype.__name__}**: {evalue}"))

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # Establish the context in which GPT will respond with role: assistant
            {
                "role": "system",
                "content": f"""You are a notebook coding assistant, designed to help users diagnose error messages. The users notebook is called "{notebook_name}". Use markdown for formatting. Rely on GitHub flavored markdown for code blocks (specifying the language for syntax highlighting). Be concise. Write code examples.""",
            },
            # The user sent code
            {"role": "user", "content": code},
            # The system literally wrote back with the error
            {"role": "system", "content": f"{etype}: {evalue}"},
            # expectation is that ChatGPT responds with:
            # { "role": "assistant", "content": ... }
        ],
    )

    # TODO: Make sure the message is from role: assistant
    display(
        Markdown(
            f"""### 💡 Suggestion
{completion["choices"][0]["message"]["content"]}
"""
        )
    )


def register(ipython=None):
    """Register the exception handler with the given IPython instance."""
    if ipython is None:
        ipython = get_ipython()

    ipython.set_custom_exc((Exception,), custom_exc)
