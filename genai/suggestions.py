"""
This module creates a custom exception handler that will send the error to OpenAI's
ChatGPT in order to help debug the code. It will also display the error in the
notebook as usual.
"""


from IPython.display import Pretty
from IPython.core.display_functions import display
from IPython import get_ipython

import openai

from traceback import TracebackException

NOTEBOOK_CODING_ASSISTANT_TEMPLATE = """You are a notebook coding assistant, designed to help users diagnose error messages.
Use markdown for formatting. Rely on GitHub flavored markdown for code blocks (specifying the language for syntax highlighting).
Be concise. Write code examples."""


# this function will be called on exceptions in any cell
def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    # still show the error within the notebook, don't just swallow it
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    try:
        # Get the current code
        code = shell.user_ns["In"][-1]

        heading = display(
            Pretty(("Let's see how we can fix this... 🔧")), display_id=True
        )

        # Highly colorized tracebacks do not help GPT as much as a clean plaintext
        # traceback.
        formatted = TracebackException(etype, evalue, tb, limit=20).format(chain=True)
        plaintext_traceback = "\n".join(formatted)

        # Cap our error report at ~1024 characters
        error_report = f"{etype.__name__}: {evalue}\n{plaintext_traceback}"

        if len(error_report) > 1024:
            error_report = error_report[:1024] + "\n..."

        messages = [
            # Establish the context in which GPT will respond with role: assistant
            {
                "role": "system",
                "content": NOTEBOOK_CODING_ASSISTANT_TEMPLATE,
            },
            # The user sent code
            {"role": "user", "content": code},
            # The system literally wrote back with the error
            {
                "role": "system",
                "content": error_report,
            },
            # expectation is that ChatGPT responds with:
            # { "role": "assistant", "content": ... }
        ]

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        # display the suggestion
        content = completion["choices"][0]["message"]["content"]
        suggestion = content

        heading.update(
            {
                "text/markdown": f"## 💡 Suggestion",
                "text/plain": "",
            },
            raw=True,
        )

        # IPython.display.Markdown() doesn't return a plaintext version so we must return a raw display for use
        # in `ipython`.

        display(
            {
                "text/plain": suggestion,
                "text/markdown": f"{suggestion}",
            },
            raw=True,
        )

    except Exception as e:
        print("Error while trying to provide a suggestion: ", e)


def register(ipython=None):
    """Register the exception handler with the given IPython instance."""
    if ipython is None:
        ipython = get_ipython()

    ipython.set_custom_exc((Exception,), custom_exc)
