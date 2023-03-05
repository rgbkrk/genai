from genai.context import get_historical_context


class FakeOutput:
    def __repr__(self):
        return "<matplotlib.axes._subplots.AxesSubplot at 0x7f8b0c0f7a90>"


def test_get_historical_context(ip):
    ip.history_manager.input_hist_raw = [
        "import pandas as pd",
        "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
        "df.plot.scatter(x='a', y='b')",
    ]

    ip.history_manager.output_hist = {
        2: FakeOutput(),
    }

    context = get_historical_context(ip)

    assert context == [
        {
            "content": "import pandas as pd",
            "role": "user",
        },
        {
            "content": "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
            "role": "user",
        },
        {
            "content": "df.plot.scatter(x='a', y='b')",
            "role": "user",
        },
        {
            "content": "<matplotlib.axes._subplots.AxesSubplot at 0x7f8b0c0f7a90>",
            "role": "system",
        },
    ]


def test_get_historical_context_with_ignore_tokens(ip):
    ip.history_manager.input_hist_raw = [
        "import pandas as pd",
        """# genai:ignore
print("This is code that should be ignored")
        """,
        "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
        "df.plot.scatter(x='a', y='b')",
    ]
    ip.history_manager.output_hist = {
        3: FakeOutput(),
    }

    context = get_historical_context(ip)

    assert context == [
        {
            "content": "import pandas as pd",
            "role": "user",
        },
        {
            "content": "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
            "role": "user",
        },
        {
            "content": "df.plot.scatter(x='a', y='b')",
            "role": "user",
        },
        {
            "content": "<matplotlib.axes._subplots.AxesSubplot at 0x7f8b0c0f7a90>",
            "role": "system",
        },
    ]


def test_get_historical_context_with_ignore_tokens_and_empty_lines(ip):
    ip.history_manager.input_hist_raw = [
        "import pandas as pd",
        """# genai:ignore
print("This is code that should be ignored")
        """,
        "",
        "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
        "df.plot.scatter(x='a', y='b')",
    ]
    ip.history_manager.output_hist = {
        4: FakeOutput(),
    }

    context = get_historical_context(ip)

    assert context == [
        {
            "content": "import pandas as pd",
            "role": "user",
        },
        {
            "content": "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
            "role": "user",
        },
        {
            "content": "df.plot.scatter(x='a', y='b')",
            "role": "user",
        },
        {
            "content": "<matplotlib.axes._subplots.AxesSubplot at 0x7f8b0c0f7a90>",
            "role": "system",
        },
    ]


def test_get_historical_context_with_ignore_tokens_and_empty_lines_and_comments(ip):
    ip.history_manager.input_hist_raw = [
        "import pandas as pd",
        """# genai:ignore
print("This is code that should be ignored")
        """,
        "",
        "# This is a comment",
        "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
        "df.plot.scatter(x='a', y='b')",
    ]
    ip.history_manager.output_hist = {
        5: FakeOutput(),
    }

    context = get_historical_context(ip)

    assert context == [
        {
            "content": "import pandas as pd",
            "role": "user",
        },
        {
            "content": "# This is a comment",
            "role": "user",
        },
        {
            "content": "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
            "role": "user",
        },
        {
            "content": "df.plot.scatter(x='a', y='b')",
            "role": "user",
        },
        {
            "content": "<matplotlib.axes._subplots.AxesSubplot at 0x7f8b0c0f7a90>",
            "role": "system",
        },
    ]
