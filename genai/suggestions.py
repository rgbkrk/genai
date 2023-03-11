"""
This module creates a custom exception handler that will send the error to OpenAI's
ChatGPT in order to help debug the code. It will also display the error in the
notebook as usual.
"""


from traceback import TracebackException

from IPython import get_ipython
from IPython.core.display_functions import display
from IPython.display import Pretty

from genai.generate import generate_exception_suggestion


def can_handle_display_updates():
    """Determine (roughly) if the client can handle display updates."""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False

        name = ipython.__class__.__name__

        if name == "ZMQInteractiveShell":
            return True
        elif name == "TerminalInteractiveShell":
            return False
        else:
            # Just assume they can otherwise
            return True
    except ImportError:
        # No IPython, so no display updates whatsoever
        return False


# this function will be called on exceptions in any cell
def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    # still show the error within the notebook, don't just swallow it
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    try:
        # Get the current code
        code = shell.user_ns["In"][-1]

        heading = display(Pretty(("Let's see how we can fix this... 🔧")), display_id=True)

        # Highly colorized tracebacks do not help GPT as much as a clean plaintext traceback.
        formatted = TracebackException(etype, evalue, tb, limit=20).format(chain=True)
        plaintext_traceback = "\n".join(formatted)

        suggestion = generate_exception_suggestion(
            code=code,
            etype=etype,
            evalue=evalue,
            plaintext_traceback=plaintext_traceback,
            stream=False,
        )

        content = ""

        for delta in suggestion:
            content += delta

            heading.update(
                {
                    "text/markdown": f"## 💡 Suggestion\n\n{content}",
                    "text/plain": f"## 💡 Suggestion\n\n{content}",
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
