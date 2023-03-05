"""
Creates user and system messages as context for ChatGPT, using the history of the current IPython session.
"""


def craft_user_message(code):
    return {
        "content": code,
        "role": "user",
    }


def craft_output_message(output):
    return {
        "content": repr(output),
        "role": "system",
    }


# tokens to idenfify which cells to ignore based on the start
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


def get_historical_context(ipython, num_messages=5):
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

    return context
