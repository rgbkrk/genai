from typing import Any, Dict, Iterator, List, TypedDict

import openai

from genai.prompts import PromptStore

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


def generate_next_from_history(
    context: List[Dict[str, str]],
    text: str,
    stream: bool = False,
) -> Iterator[str]:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # Establish the context of the conversation
            {
                "role": "system",
                "content": PromptStore.assist_prompt,
            },
            # Presumably In, Out
            *context,
            # The user's code or request
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

    # Just in case, cap the error report
    if len(error_report) > 1024:
        error_report = error_report[:1024] + "\n..."

    messages = []

    # Establish the role for ChatGPT
    messages.append(
        {
            "role": "system",
            "content": PromptStore.exception_prompt,
        },
    )

    # User executed code
    if code is not None:
        messages.append({"role": "user", "content": code})

    # Code created an error in the system
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
