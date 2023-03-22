"""
This module creates a custom exception handler that will send the error to OpenAI's
ChatGPT in order to help debug the code. It will also display the error in the
notebook as usual.
"""

from traceback import TracebackException
from types import TracebackType
from typing import Type

from IPython import InteractiveShell, get_ipython

from genai.context import PastAssists, PastErrors
from genai.display import GenaiMarkdown, Stage, can_handle_display_updates
from genai.generate import generate_exception_suggestion


# this function will be called on exceptions in any cell
def custom_exc(
    shell: "InteractiveShell",
    etype: Type[BaseException],
    evalue: BaseException,
    tb: TracebackType,
    tb_offset=None,
):
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

        gm = GenaiMarkdown(
            "Let's see how we can fix this... 🔧",
            stage=Stage.STARTING,
        )
        gm.display()

        # Highly colorized tracebacks do not help GPT as much as a clean plaintext traceback.
        formatted = TracebackException(etype, evalue, tb, limit=3).format(chain=True)
        plaintext_traceback = "\n".join(formatted)

        # Track context for future suggestions
        PastErrors.add(execution_count, etype, evalue, tb)
        PastAssists.add(execution_count, gm)

        stream = can_handle_display_updates()

        suggestion = generate_exception_suggestion(
            code=code,
            etype=etype,
            evalue=evalue,
            plaintext_traceback=plaintext_traceback,
            stream=stream,
        )

        gm.stage = Stage.GENERATING

        gm.message = "## 💡 Suggestion\n"
        gm.consume(suggestion)
        gm.stage = Stage.FINISHED

    except Exception as e:
        print("Error while trying to provide a suggestion: ", e)
    except KeyboardInterrupt:
        # If we have our heading, replace it with empty text and take out any stage information
        if "gm" in locals():
            gm.message = " "
            gm.stage = None


def register(ipython=None):
    """Register the exception handler with the given IPython instance."""
    if ipython is None:
        ipython = get_ipython()

    ipython.set_custom_exc((Exception,), custom_exc)
