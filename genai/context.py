"""
Creates user and system messages as context for ChatGPT, using the history of the current IPython session.
"""
from typing import Any, Dict, Optional

try:
    import pandas as pd

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False


def craft_message(text: str, role: str = "user") -> Dict[str, str]:
    return {"content": text, "role": role}


def craft_user_message(code: str) -> Dict[str, str]:
    return craft_message(code, "user")


def craft_output_message(output: Any) -> Dict[str, str]:
    """Craft a message from the output of an execution."""
    return craft_message(repr_genai(output), "system")


class Context:
    def __init__(self):
        self._context = []

    def append(self, text: str, execution_count: Optional[int] = None, role: str = "user"):
        contextual_message = {
            "message": craft_message(text, role=role),
            "execution_count": execution_count,
        }
        self._context.append(contextual_message)

    @property
    def messages(self):
        return [message["message"] for message in self._context]


def repr_genai_pandas(output: Any) -> str:
    if not PANDAS_INSTALLED:
        return repr(output)

    if isinstance(output, pd.DataFrame):
        # to_markdown() does not use the max_rows and max_columns options
        # so we have to truncate the dataframe ourselves
        num_columns = min(pd.options.display.max_columns, output.shape[1])
        num_rows = min(pd.options.display.max_rows, output.shape[0])
        sampled = output.sample(num_columns, axis=1).sample(num_rows, axis=0)
        return sampled.to_markdown()

    if isinstance(output, pd.Series):
        # Similar truncation for series
        num_rows = min(pd.options.display.max_rows, output.shape[0])
        sampled = output.sample(num_rows)
        return sampled.to_markdown()

    return repr(output)


def repr_genai(output: Any) -> str:
    '''Compute a GPT-3.5 friendly representation of the output of a cell.

    For DataFrames and Series this means Markdown.
    '''
    if not PANDAS_INSTALLED:
        return repr(output)

    with pd.option_context(
        'display.max_rows', 5, 'display.html.table_schema', False, 'display.max_columns', 20
    ):
        return repr_genai_pandas(output)


# tokens to idenfify which cells to ignore based on the first line
ignore_tokens = [
    "# genai:ignore",
    "#ignore",
    "# ignore",
    "%%assist",
    "get_ipython",
    "%load_ext",
    "import genai",
    "%pip install",
    "#%%assist",
]


def build_context(history_manager, start=1, stop=None):
    context = Context()

    for (session, execution_counter, cell_text) in history_manager.get_range(
        session=0, start=start, stop=stop
    ):
        print('session', session)
        if any(cell_text.startswith(token) for token in ignore_tokens):
            continue

        context.append(cell_text, role="user", execution_count=execution_counter)

        output = history_manager.output_hist.get(execution_counter)
        if output is not None:
            context.append(repr_genai(output), role="system", execution_count=execution_counter)

    return context
