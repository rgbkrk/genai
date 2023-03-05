import sys
from unittest import mock

from genai import suggestions

from IPython.display import HTML


def test_register():
    fake_ip = mock.MagicMock()
    suggestions.register(fake_ip)

    fake_ip.set_custom_exc.assert_called_once_with((Exception,), suggestions.custom_exc)


@mock.patch(
    "IPython.core.display_functions.display",
    autospec=True,
)
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
def test_custom_exc(create, display, ip):
    try:
        raise Exception("this is just a test")
    except Exception:
        (etype, evalue, tb) = sys.exc_info()

    ip.showtraceback = mock.MagicMock()

    ip.user_ns["In"] = ["import pandas as pd", "fancy code"]

    suggestions.custom_exc(ip, etype, evalue, tb, tb_offset=None)

    # Ensure that we always report the users error back via the shell's showtraceback
    ip.showtraceback.assert_called_once_with((etype, evalue, tb), tb_offset=None)

    create.assert_called_once()

    args, kwargs = create.call_args

    assert args == ()
    assert kwargs["model"] == "gpt-3.5-turbo"
    assert len(kwargs["messages"]) == 3
    assert kwargs["messages"][0] == {
        "role": "system",
        "content": suggestions.NOTEBOOK_CODING_ASSISTANT_TEMPLATE,
    }
    assert kwargs["messages"][1] == {
        "role": "user",
        "content": "fancy code",
    }
    assert kwargs["messages"][2]["role"] == "system"
    assert kwargs["messages"][2]["content"].startswith("Exception: this is just a test")

    # display will be called *lots* of times
    display.assert_called()

    # The last call to display should be the heading
    heading = display.call_args_list[-1][0][0]
    assert heading.data == "<h3>Here's a way to fix this 🛠</h3>"
