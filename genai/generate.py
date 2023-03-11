import openai

NOTEBOOK_CREATE_NEXT_CELL_PROCLAMATION = """
As a coding assistant, your task is to help users write code in Python within Jupyter Notebooks. Provide comments and code for the user to read and edit, ensuring it can be run successfully. The user will be able to run the code in the cell and see the output.
""".strip()  # noqa: E501


NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION = """
As a coding assistant, you'll diagnose errors in Python code written in a Jupyter Notebook. Use %pip instead of !pip and format your response using GitHub flavored markdown. Provide concise code examples in your response which will be rendered in Markdown in the notebook.
""".strip()  # noqa: E501


def content(completion):
    return completion["choices"][0]["message"]["content"]


def deltas(completion):
    for chunk in completion:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]


def generate_next_cell(
    context,  # List[Dict[str, str]]
    text,
    stream=False,
):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # Establish the context in which GPT will respond
            {
                "role": "system",
                "content": NOTEBOOK_CREATE_NEXT_CELL_PROCLAMATION,
            },
            # In, Out
            *context,
            # The user code/text
            {
                "role": "user",
                "content": text,
            },
        ],
        stream=stream,
    )

    if stream:
        yield from deltas(response)
    else:
        yield content(response)


def generate_exception_suggestion(
    # The user's code
    code,
    # The exception with traceback
    etype,
    evalue,
    plaintext_traceback,
    stream=False,
):
    # Cap our error report at ~1024 characters
    error_report = f"{etype.__name__}: {evalue}\n{plaintext_traceback}"

    if len(error_report) > 1024:
        error_report = error_report[:1024] + "\n..."

    messages = [
        # Establish the context in which GPT will respond with role: assistant
        {
            "role": "system",
            "content": NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION,
        },
        # The user sent code
        {"role": "user", "content": code},
        # The system literally wrote back with the error
        {
            "role": "system",
            "content": error_report,
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=stream,
    )

    if stream:
        yield from deltas(response)
    else:
        yield content(response)
