import sys
from unittest import mock

from genai import generate


def mock_deltas():
    yield {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "Here's",
                },
            },
        ]
    }

    yield {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "Something",
                },
            },
        ]
    }


@mock.patch(
    "openai.ChatCompletion.create",
    return_value=mock_deltas(),
    autospec=True,
)
def test_generate_exception_suggestion(create):
    try:
        raise Exception("this is just a test")
    except Exception:
        (etype, evalue, tb) = sys.exc_info()

    response = generate.generate_exception_suggestion(
        code="fancy code",
        etype=etype,
        evalue=evalue,
        plaintext_traceback="""Traceback (most recent call last):
        <<snip>>
        """,
        stream=True,
    )

    for buh in response:
        assert buh in ["Here's", "Something"]
