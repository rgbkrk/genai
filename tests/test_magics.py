from unittest import mock

from genai import generate


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
                "content": generate.NOTEBOOK_CODING_ASSISTANT_TEMPLATE,
            },
            {
                "role": "user",
                "content": "create a scatterplot from df",
            },
        ],
    )
