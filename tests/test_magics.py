from unittest import mock

from genai.prompts import PromptStore
from genai.context import PastAssists


@mock.patch(
    "openai.ChatCompletion.create",
    return_value={
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Here's a suggestion",
                },
            },
        ],
    },
    autospec=True,
)
def test_assist_magic(create, ip):
    ip.run_cell_magic(magic_name="assist", line="", cell="create a scatterplot from df")

    # Check that create was called with the correct arguments
    create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": PromptStore.assist_prompt,
            },
            {
                "role": "user",
                "content": "create a scatterplot from df",
            },
        ],
        stream=False,
    )


@mock.patch(
    "openai.ChatCompletion.create",
    return_value={
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "just like code better",
                },
            },
        ],
    },
    autospec=True,
)
def test_assist_magic_with_args(create, ip):
    ip.run_cell_magic(
        magic_name="assist",
        line="--verbose",
        cell="create a scatterplot from df",
    )
    execution_count = ip.execution_count

    # Check that create was called with the correct arguments
    create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": PromptStore.assist_prompt,
            },
            {
                "role": "user",
                "content": "create a scatterplot from df",
            },
        ],
        stream=False,
    )

    # We can look at past assists to see what the assistant has suggested
    assist = PastAssists.get(execution_count)

    assert assist.message == "just like code better"


@mock.patch(
    "openai.ChatCompletion.create",
    return_value={
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "superplot(df)",
                },
            },
        ],
    },
    autospec=True,
)
def test_assist_magic_with_fresh_arg(create, ip):
    ip.run_cell_magic(
        magic_name="assist",
        line="--fresh",
        cell="create a scatterplot from df",
    )

    create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": PromptStore.assist_prompt,
            },
            # Note that there is zero other context, due to running with --fresh
            {
                "role": "user",
                "content": "create a scatterplot from df",
            },
        ],
        stream=False,
    )

    # We can look at past assists to see what the assistant has suggested
    assist = PastAssists.get(ip.execution_count)

    assert assist.message == "superplot(df)"
