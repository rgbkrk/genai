from unittest.mock import MagicMock, call, patch

import pandas as pd

from genai.context import get_historical_context, repr_genai, repr_genai_pandas


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


@patch("pandas.DataFrame.sample", return_value=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
def test_get_historical_context_pandas(sample, ip):
    ip.history_manager.input_hist_raw = [
        "import pandas as pd",
        "df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})",
        "df",
    ]

    with pd.option_context("display.max_rows", 10, "display.max_columns", 10):
        ip.history_manager.output_hist = {
            2: pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        }

        context = get_historical_context(ip)

    # First call should be on columns with axis=1
    sample.assert_has_calls(
        [
            call(2, axis=1),
            call(3, axis=0),
        ]
    )

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
            "content": "df",
            "role": "user",
        },
        {
            "content": '|    |   a |   b |\n|---:|----:|----:|\n|  0 |   1 |   4 |\n|  1 |   2 |   5 |\n|  2 |   3 |   6 |',
            "role": "system",
        },
    ]


def test_repr_genai_pandas():
    # create a mock DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # create a MagicMock for the sample method
    mock_sample = MagicMock(return_value=df)
    # assign the MagicMock to the sample attribute of the DataFrame
    df.sample = mock_sample

    # call the function with the mock DataFrame
    result = repr_genai_pandas(df)

    # check that the MagicMock was called with the correct arguments
    mock_sample.assert_called_with(min(pd.options.display.max_rows, df.shape[0]), axis=0)

    # check that the result of the function is the same as the expected result
    expected = df.sample(min(pd.options.display.max_rows, df.shape[0]), axis=0).to_markdown()
    assert result == expected


@patch("pandas.Series.sample", return_value=pd.Series([1, 2, 3]))
def test_repr_genai_pandas_series(sample, ip):
    # create a mock Series
    series = pd.Series([1, 2, 3])
    # call the function with the mock DataFrame
    result = repr_genai_pandas(series)

    # check that the MagicMock was called with the correct arguments
    sample.assert_called_with(3)

    # check that the result of the function is the same as the expected result
    expected = series.sample(
        min(pd.options.display.max_rows, series.shape[0]), axis=0
    ).to_markdown()
    assert result == expected


def test_repr_genai_pandas_not_series_or_dataframe():
    output = "hello"
    result = repr_genai_pandas(output)
    assert result == "'hello'"


@patch("genai.context.PANDAS_INSTALLED", False)
def test_repr_genai_without_pandas():
    output = [1, 2, 3]
    result = repr_genai_pandas(output)
    assert result == "[1, 2, 3]"

    result = repr_genai(output)
    assert result == "[1, 2, 3]"
