"""
This module creates a custom exception handler that will send the error to OpenAI's
ChatGPT in order to help debug the code. It will also display the error in the
notebook as usual.
"""

from enum import Enum
from traceback import TracebackException

from IPython import get_ipython
from IPython.core.display_functions import display

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


class Stage(str, Enum):
    """The stage of feedback generation"""

    STARTING = "starting"
    GENERATING = "generating"
    FINISHED = "finished"


def GenaiMarkdown(text, stage=None):
    return (
        {
            "text/markdown": text,
            "text/plain": text,
        },
        {
            "text/markdown": {
                "genai": {
                    "stage": stage,
                }
            }
        },
    )


# this function will be called on exceptions in any cell
def custom_exc(shell, etype, evalue, tb, tb_offset=None):
    # still show the error within the notebook, don't just swallow it
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    # On interrupt, just let it be
    if etype == KeyboardInterrupt:
        return
    # On exit, release the user
    elif etype == SystemExit:
        return

    try:
        code = None

        execution_count = shell.execution_count

        In = shell.user_ns["In"]
        history_manager = shell.history_manager

        # If the history is available, use that as it has the raw inputs (including magics)
        if (history_manager is not None) and (
            execution_count == len(history_manager.input_hist_raw) - 1
        ):
            code = shell.history_manager.input_hist_raw[execution_count]
        # Fallback on In
        elif In is not None and execution_count == len(In) - 1:
            # Otherwise, use the current input buffer
            code = In[execution_count]
        # Otherwise history may not have been stored (store_history=False), so we should not send the
        # code to GPT.
        else:
            code = None

        data, metadata = GenaiMarkdown("Let's see how we can fix this... 🔧", stage=Stage.STARTING)
        heading = display(data, metadata=metadata, raw=True, display_id=True)

        # Highly colorized tracebacks do not help GPT as much as a clean plaintext traceback.
        formatted = TracebackException(etype, evalue, tb, limit=20).format(chain=True)
        plaintext_traceback = "\n".join(formatted)

        stream = can_handle_display_updates()

        suggestion = generate_exception_suggestion(
            code=code,
            etype=etype,
            evalue=evalue,
            plaintext_traceback=plaintext_traceback,
            stream=stream,
        )

        content = ""

        for delta in suggestion:
            content += delta

            data, metadata = GenaiMarkdown(f"## 💡 Suggestion\n\n{content}", stage=Stage.GENERATING)

            heading.update(
                data,
                metadata=metadata,
                raw=True,
            )

        data, metadata = GenaiMarkdown(f"## 💡 Suggestion\n\n{content}", stage=Stage.FINISHED)
        heading.update(data, metadata=metadata, raw=True)

    except Exception as e:
        print("Error while trying to provide a suggestion: ", e)
    except KeyboardInterrupt:
        # If we have the display heading, we can update it with empty text.
        if "heading" in locals():
            heading.update(
                {
                    "text/markdown": "",
                    "text/plain": "",
                },
                raw=True,
            )


def register(ipython=None):
    """Register the exception handler with the given IPython instance."""
    if ipython is None:
        ipython = get_ipython()

    ipython.set_custom_exc((Exception,), custom_exc)
