import pytest
from IPython.testing.globalipapp import start_ipython

from unittest import mock

import openai

from genai import generate


@pytest.fixture(scope="session")
def session_ip():
    yield start_ipython()


@pytest.fixture(scope="function")
def ip(session_ip):
    session_ip.run_line_magic(magic_name="load_ext", line="genai")
    yield session_ip
    # Include unload_ext once we define how to unload our extension
    # session_ip.run_line_magic(magic_name="unload_ext", line="genai")
    session_ip.run_line_magic(magic_name="reset", line="-f")


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
