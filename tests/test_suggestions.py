import sys
from unittest import mock

from genai import suggestions, generate

from genai.suggestions import can_handle_display_updates


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

    ip.execution_count = 2
    ip.user_ns["In"] = None  # Ensure we use history_manager
    ip.history_manager.input_hist_raw = ["", "import pandas as pd", "fancy code"]
    print("history", ip.history_manager.input_hist_raw)

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
                    "content": "Here's a suggestion",
                },
            },
        ],
    },
    autospec=True,
)
def test_custom_exc_fallback_on_In(create, publish_display_data, display, ip):
    try:
        raise Exception("this is just a test")
    except Exception:
        (etype, evalue, tb) = sys.exc_info()

    ip.showtraceback = mock.MagicMock()

    ip.execution_count = 2
    ip.user_ns["In"] = ["", "import pandas as pd", "fancy code"]
    ip.history_manager.input_hist_raw = []

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

    ip.execution_count = 2
    print('execution_count!!!', ip.execution_count)
    ip.user_ns["In"] = ["", "import pandas as pd", "fancy code"]
    ip.history_manager.input_hist_raw = ["", "import pandas as pd", "fancy code"]
    print("history", ip.history_manager.input_hist_raw)

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


@mock.patch(
    "IPython.core.display_functions.display",
    autospec=True,
)
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
def test_custom_exc_error_inside(create, print, display, ip):
    ip.showtraceback = mock.MagicMock()

    try:
        raise Exception("this is just a test")
    except Exception:
        (etype, evalue, tb) = sys.exc_info()

    # Force an error inside the custom_exc function
    display.side_effect = TypeError("'Fun' is not allowed")
    suggestions.custom_exc(ip, etype, evalue, tb, tb_offset=None)

    # Ensure that we always report the users error back via the shell's showtraceback
    ip.showtraceback.assert_called_once_with((etype, evalue, tb), tb_offset=None)

    assert print.call_args.args[0] == "Error while trying to provide a suggestion: "
    assert print.call_args.args[1].__class__ == TypeError
    assert print.call_args.args[1].args[0] == "'Fun' is not allowed"


@mock.patch(
    "IPython.core.display_functions.display",
    autospec=True,
)
@mock.patch(
    "IPython.core.display_functions.publish_display_data",
)
def test_pass_interrupts_through(publish_display_data, display, ip):
    ip.showtraceback = mock.MagicMock()

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        (etype, evalue, tb) = sys.exc_info()

    suggestions.custom_exc(ip, etype, evalue, tb, tb_offset=None)

    # Ensure that we always report the users error back via the shell's showtraceback
    ip.showtraceback.assert_called_once_with((etype, evalue, tb), tb_offset=None)

    # display will be called *lots* of times
    # For some reason this is only those that are using display IDs
    display.assert_not_called()
    # We also have to look at the publish_display_data calls
    publish_display_data.assert_not_called()


@mock.patch(
    "IPython.core.display_functions.display",
    autospec=True,
)
@mock.patch(
    "IPython.core.display_functions.publish_display_data",
)
def test_pass_system_exit_through(publish_display_data, display, ip):
    ip.showtraceback = mock.MagicMock()

    try:
        raise SystemExit()
    except SystemExit:
        (etype, evalue, tb) = sys.exc_info()

    suggestions.custom_exc(ip, etype, evalue, tb, tb_offset=None)

    # Ensure that we always report the users error back via the shell's showtraceback
    ip.showtraceback.assert_called_once_with((etype, evalue, tb), tb_offset=None)

    # display will be called *lots* of times
    # For some reason this is only those that are using display IDs
    display.assert_not_called()
    # We also have to look at the publish_display_data calls
    publish_display_data.assert_not_called()


def fake_IPython(name):
    '''Create a fake named IPython shell'''
    return mock.MagicMock(
        get_ipython=mock.Mock(return_value=mock.Mock(__class__=mock.Mock(__name__=name)))
    )


def test_can_handle_display_updates_with_ZMQInteractiveShell():
    with mock.patch('builtins.__import__', return_value=fake_IPython("ZMQInteractiveShell")):
        assert can_handle_display_updates() is True


def test_can_handle_display_updates_with_TerminalInteractiveShell():
    with mock.patch('builtins.__import__', return_value=fake_IPython("TerminalInteractiveShell")):
        assert can_handle_display_updates() is False


def test_no_IPython_means_no_display_updates():
    with mock.patch('builtins.__import__', side_effect=ImportError()):
        assert can_handle_display_updates() is False


def test_can_handle_display_updates_with_other_shell():
    with mock.patch('builtins.__import__', return_value=fake_IPython("SuperCoolInteractiveShell")):
        assert can_handle_display_updates() is True
