"""
Creates user and system messages as context for ChatGPT, using the history of the current IPython session.
"""
from traceback import TracebackException
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

from genai.display import GenaiMarkdown

try:
    import numpy as np
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


def summarize_dataframe(df, sample_rows=5, sample_columns=20):
    """
    Create a summary of a Pandas DataFrame for ChatGPT.

    Parameters:
        df (Pandas DataFrame): The dataframe to be summarized.
        sample_rows (int): The number of rows to sample
        sample_columns (int): The number of columns to sample

    Returns:
        A markdown string with a summary of the dataframe
    """
    num_rows, num_cols = df.shape

    # Column Summary
    ## Missing value summary for all columns
    missing_values = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    missing_values['% Missing'] = missing_values['Missing Values'] / num_rows * 100

    ## Data type summary for all columns
    column_info = pd.concat([df.dtypes, missing_values], axis=1).reset_index()
    column_info.columns = ["Column Name", "Data Type", "Missing Values", "% Missing"]
    column_info['Data Type'] = column_info['Data Type'].astype(str)

    # Basic summary statistics for numerical and categorical columns
    # get basic statistical information for each column
    numerical_summary = (
        df.describe(include=[np.number]).T.reset_index().rename(columns={'index': 'Column Name'})
    )

    has_categoricals = any(df.select_dtypes(include=['category', 'datetime', 'timedelta']).columns)

    if has_categoricals:
        categorical_describe = df.describe(include=['category', 'datetime', 'timedelta'])
        categorical_summary = categorical_describe.T.reset_index().rename(
            columns={'index': 'Column Name'}
        )
    else:
        categorical_summary = pd.DataFrame(columns=['Column Name'])

    sample_columns = min(sample_columns, df.shape[1])
    sample_rows = min(sample_rows, df.shape[0])
    sampled = df.sample(sample_columns, axis=1).sample(sample_rows, axis=0)

    tablefmt = "github"

    # create the markdown string for output
    output = (
        f"## Dataframe Summary\n\n"
        f"Number of Rows: {num_rows:,}\n\n"
        f"Number of Columns: {num_cols:,}\n\n"
        f"### Column Information\n\n{column_info.to_markdown(tablefmt=tablefmt)}\n\n"
        f"### Numerical Summary\n\n{numerical_summary.to_markdown(tablefmt=tablefmt)}\n\n"
        f"### Categorical Summary\n\n{categorical_summary.to_markdown(tablefmt=tablefmt)}\n\n"
        f"### Sample Data ({sample_rows}x{sample_columns})\n\n{sampled.to_markdown(tablefmt=tablefmt)}"
    )

    return output


def summarize_series(series, sample_size=5):
    """
    Create a summary of a Pandas Series for ChatGPT.

    Parameters:
        series (pd.Series): The series to be summarized.
        sample_size (int): The number of values to sample

    Returns:
        A markdown string with a summary of the series
    """
    # Get basic series information
    num_values = len(series)
    data_type = series.dtype
    num_missing = series.isnull().sum()
    percent_missing = num_missing / num_values * 100

    # Get summary statistics based on the data type
    if np.issubdtype(data_type, np.number):
        summary_statistics = series.describe().to_frame().T
    elif pd.api.types.is_string_dtype(data_type):
        summary_statistics = series.describe(datetime_is_numeric=True).to_frame().T
    else:
        summary_statistics = series.describe().to_frame().T

    # Sample data
    sampled = series.sample(min(sample_size, num_values))

    tablefmt = "github"

    # Create the markdown string for output
    output = (
        f"## Series Summary\n\n"
        f"Number of Values: {num_values:,}\n\n"
        f"Data Type: {data_type}\n\n"
        f"Missing Values: {num_missing:,} ({percent_missing:.2f}%)\n\n"
        f"### Summary Statistics\n\n{summary_statistics.to_markdown(tablefmt=tablefmt)}\n\n"
        f"### Sample Data ({sample_size})\n\n{sampled.to_frame().to_markdown(tablefmt=tablefmt)}"
    )

    return output


def repr_genai_pandas(output: Any) -> str:
    if not PANDAS_INSTALLED:
        return repr(output)

    if isinstance(output, pd.DataFrame):
        num_columns = min(pd.options.display.max_columns, output.shape[1])
        num_rows = min(pd.options.display.max_rows, output.shape[0])
        return summarize_dataframe(output, sample_rows=num_rows, sample_columns=num_columns)

    if isinstance(output, pd.Series):
        num_rows = min(pd.options.display.max_rows, output.shape[0])
        return summarize_series(output, sample_size=num_rows)

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
    "#genai:ignore",
    "#ignore",
    "# ignore",
    "get_ipython",
    "%load_ext",
    "%pip install",
    "%%prompt",
]


class PastErrors:
    """Tracks previous errors in the session"""

    errors: Dict[str, str] = {}

    @classmethod
    def add(
        cls,
        execution_count: int,
        etype: Type[BaseException],
        evalue: BaseException,
        tb: TracebackType,
    ):
        condensed_error = "\n".join(
            TracebackException(etype, evalue, tb, limit=2).format(chain=True)
        )
        cls.errors[str(execution_count)] = condensed_error

    @classmethod
    def clear(cls):
        cls.errors = {}

    @classmethod
    def get(cls, execution_count: Union[int, str]) -> Optional[str]:
        return cls.errors.get(str(execution_count))


class PastAssists:
    """Tracks previous assists in the session"""

    assists: Dict[str, GenaiMarkdown] = {}

    @classmethod
    def add(cls, execution_count: int, assist: GenaiMarkdown):
        cls.assists[str(execution_count)] = assist

    @classmethod
    def clear(cls):
        cls.assists = {}

    @classmethod
    def get(cls, execution_count: Union[int, str]) -> Optional[GenaiMarkdown]:
        return cls.assists.get(str(execution_count))


class Context:
    '''Utility class to build the context for ChatGPT from an IPython session'''

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


def build_context(history_manager, start=1, stop=None):
    context = Context()

    for session, execution_counter, cell_text in history_manager.get_range(
        session=0, start=start, stop=stop
    ):
        if any(cell_text.startswith(token) for token in ignore_tokens):
            continue

        # User Code `In[*]:`
        context.append(cell_text, role="user", execution_count=execution_counter)

        # System Error Output
        past_error = PastErrors.get(execution_counter)
        if past_error is not None:
            context.append(past_error, role="system", execution_count=execution_counter)

        # Assistant Output
        past_assist = PastAssists.get(execution_counter)
        if past_assist is not None:
            context.append(past_assist.message, role="assistant", execution_count=execution_counter)

        # System Outputs `Out[*]:`
        output = history_manager.output_hist.get(execution_counter)
        if output is not None:
            context.append(repr_genai(output), role="system", execution_count=execution_counter)

    return context
