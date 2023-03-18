"""
Creates user and system messages as context for ChatGPT, using the history of the current IPython session.
"""

try:
    import pandas as pd

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

from . import tokens


def craft_message(text, role="user"):
    return {"content": text, "role": role}


def craft_user_message(code):
    return craft_message(code, "user")


def repr_genai_pandas(output):
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


def repr_genai(output):
    '''Compute a GPT-3.5 friendly representation of the output of a cell.

    For DataFrames and Series this means Markdown.
    '''
    if not PANDAS_INSTALLED:
        return repr(output)

    with pd.option_context(
        'display.max_rows', 5, 'display.html.table_schema', False, 'display.max_columns', 20
    ):
        return repr_genai_pandas(output)


def craft_output_message(output):
    """Craft a message from the output of a cell."""
    return craft_message(repr_genai(output), "system")


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


def get_historical_context(ipython, num_messages=5, model="gpt-3.5-turbo-0301"):
    """Create a series of messages to use as context for ChatGPT."""
    raw_inputs = ipython.history_manager.input_hist_raw

    # Now filter out any inputs that start with our filters
    # This has to keep the input index as the key for the output
    inputs = {}
    for i, input in enumerate(raw_inputs):
        if input is None or input.strip() == "":
            continue

        if not any(input.startswith(token) for token in ignore_tokens):
            inputs[i] = input

    outputs = ipython.history_manager.output_hist

    indices = sorted(inputs.keys())
    context = []

    # We will use the last `num_messages` inputs and outputs to establish context
    for index in indices[-num_messages:]:
        context.append(craft_user_message(inputs[index]))

        if index in outputs:
            context.append(craft_output_message(outputs[index]))

    context = tokens.trim_messages_to_fit_token_limit(context, model=model)

    return context
