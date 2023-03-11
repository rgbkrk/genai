import sys
from unittest import mock

from genai import suggestions, generate


def test_register():
    fake_ip = mock.MagicMock()
    suggestions.register(fake_ip)

    fake_ip.set_custom_exc.assert_called_once_with((Exception,), suggestions.custom_exc)


@mock.patch(
    "IPython.core.display_functions.display",
    autospec=True,
)
@mock.patch(
    "IPython.core.display_functions.publish_display_data",
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
def test_custom_exc(create, publish_display_data, display, ip):
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
        "content": generate.NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION,
    }
    assert kwargs["messages"][1] == {
        "role": "user",
        "content": "fancy code",
    }
    assert kwargs["messages"][2]["role"] == "system"
    assert kwargs["messages"][2]["content"].startswith("Exception: this is just a test")

    # display will be called *lots* of times
    # For some reason this is only those that are using display IDs
    display.assert_called()
    # We also have to look at the publish_display_data calls
    publish_display_data.assert_called()

    last_call = display.call_args_list[-1]

    # get kwargs from the last call
    args = last_call[0]
    kwargs = last_call[1]

    # Now to check that we're publishing the correct data from the API
    assert args[0] == {
        "text/markdown": "## 💡 Suggestion\n\nHere's a suggestion",
        "text/plain": "## 💡 Suggestion\n\nHere's a suggestion",
    }


@mock.patch(
    "IPython.core.display_functions.display",
    autospec=True,
)
@mock.patch(
    "IPython.core.display_functions.publish_display_data",
)
@mock.patch(
    "openai.ChatCompletion.create",
    return_value={
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "When you hear 'aaaaaa' you know it's the intro to a 90s Nickolodeon show about monsters.",
                },
            },
        ],
    },
    autospec=True,
)
def test_custom_exc_long_traceback(create, publish_display_data, display, ip):
    try:
        raise Exception("a" * 2000)
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
        "content": generate.NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION,
    }
    assert kwargs["messages"][1] == {
        "role": "user",
        "content": "fancy code",
    }

    error_message = kwargs["messages"][2]
    assert error_message["role"] == "system"
    assert error_message["content"].startswith("Exception: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert error_message["content"].endswith("\n...")

    # Our max length plus the newlined ellipsis
    assert len(error_message["content"]) == 1024 + 4

    # display will be called *lots* of times
    # For some reason this is only those that are using display IDs
    display.assert_called()
    # We also have to look at the publish_display_data calls
    publish_display_data.assert_called()

    last_call = display.call_args_list[-1]

    # get kwargs from the last call
    args = last_call[0]
    kwargs = last_call[1]

    # Now to check that we're publishing the correct data from the API
    assert args[0] == {
        "text/markdown": "## 💡 Suggestion\n\nWhen you hear 'aaaaaa' you know it's the intro to a 90s Nickolodeon show about monsters.",
        "text/plain": "## 💡 Suggestion\n\nWhen you hear 'aaaaaa' you know it's the intro to a 90s Nickolodeon show about monsters.",
    }


@mock.patch("builtins.print")
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
def test_custom_exc_error_inside(create, print, ip):
    ip.showtraceback = mock.MagicMock()

    try:
        raise Exception("this is just a test")
    except Exception:
        (etype, evalue, tb) = sys.exc_info()

    # Force a TypeError in the suggestion code
    # To make sure we exercise the fallback
    ip.user_ns["In"] = None
    suggestions.custom_exc(ip, etype, evalue, tb, tb_offset=None)

    # Ensure that we always report the users error back via the shell's showtraceback
    ip.showtraceback.assert_called_once_with((etype, evalue, tb), tb_offset=None)

    assert print.call_args.args[0] == "Error while trying to provide a suggestion: "
    assert print.call_args.args[1].__class__ == TypeError
    assert print.call_args.args[1].args[0] == "'NoneType' object is not subscriptable"
