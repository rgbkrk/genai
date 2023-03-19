from typing import Any, Dict, Iterator, List, TypedDict

import openai

NOTEBOOK_CREATE_NEXT_CELL_PROCLAMATION = """
As a coding assistant, your task is to help users write code in Python within Jupyter Notebooks. Provide comments and code for the user to read and edit, ensuring it can be run successfully. The user will be able to run the code in the cell and see the output.
""".strip()  # noqa: E501


NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION = """
As a coding assistant, you'll diagnose errors in Python code written in a Jupyter Notebook. Format your response using markdown. Making sure to include the language around code blocks, like

```python
# code
```

Provide concise code examples in your response which will be rendered in Markdown in the notebook. The user will not be able to respond to your response.
""".strip()  # noqa: E501


Completion = TypedDict(
    "Completion",
    {
        "choices": List[Dict[str, Any]],
    },
)


def content(completion: Completion):
    return completion["choices"][0]["message"]["content"]


Delta = TypedDict(
    "Delta",
    {
        "content": str,
    },
)


StreamChoice = TypedDict(
    "StreamChoice",
    {
        "delta": Delta,
    },
)

StreamCompletion = TypedDict(
    "StreamCompletion",
    {
        "choices": List[StreamChoice],
    },
)


def deltas(completion: Iterator[StreamCompletion]) -> Iterator[str]:
    for chunk in completion:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            yield delta["content"]


def generate_next_cell(
    context: List[Dict[str, str]],
    text: str,
    stream: bool = False,
) -> Iterator[str]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": NOTEBOOK_CREATE_NEXT_CELL_PROCLAMATION,
            },
            *context,
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
    code: str,
    etype: type,
    evalue: BaseException,
    plaintext_traceback: str,
    stream: bool = False,
) -> Iterator[str]:
    error_report = f"{etype.__name__}: {evalue}\n{plaintext_traceback}"

    if len(error_report) > 1024:
        error_report = error_report[:1024] + "\n..."

    messages = []

    messages.append(
        {
            "role": "system",
            "content": NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION,
        },
    )

    if code is not None:
        messages.append({"role": "user", "content": code})

    messages.append(
        {
            "role": "system",
            "content": error_report,
        },
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=stream,
    )

    if stream:
        yield from deltas(response)
    else:
        yield content(response)
