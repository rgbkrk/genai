from unittest.mock import MagicMock, call, patch

import pandas as pd

from genai.context import build_context, repr_genai, repr_genai_pandas, PastAssists, PastErrors
from genai.display import GenaiMarkdown


def test_past_errors():
    # Test adding and getting errors
    PastErrors.add(1, ValueError, ValueError("Test error"), None)
    error = PastErrors.get(1)
    assert "Test error" in error
    assert PastErrors.get(2) is None

    # Test clearing errors
    PastErrors.clear()
    assert PastErrors.get(1) is None


def test_past_assists():
    # Test adding and getting assists
    assist_md = GenaiMarkdown("Test assist")
    PastAssists.add(1, assist_md)
    assert PastAssists.get(1) == assist_md
    assert PastAssists.get(2) is None

    # Test clearing assists
    PastAssists.clear()
    assert PastAssists.get(1) is None


def test_build_context_empty_history(ip):
    # Test build_context with no history
    context = build_context(ip.history_manager)

    assert context.messages == []


def test_build_context_single_input_output(ip):
    # Test build_context with a single input and output
    ip.run_cell("a = 1", store_history=True)
    ip.run_cell("a + 1", store_history=True)

    context = build_context(ip.history_manager)

    assert len(context.messages) == 3
    assert context.messages[0] == {"content": "a = 1", "role": "user"}
    assert context.messages[1]["role"] == "user"
    assert context.messages[1]["content"] == "a + 1"
    assert context.messages[2]["role"] == "system"
    assert context.messages[2]["content"].startswith("2")


def test_build_context_ignore_tokens(ip):
    # Test build_context ignoring inputs with specific tokens
    ip.run_cell("#ignore\nimport time\ntime.sleep(0)", store_history=True)
    ip.run_cell("a = 1", store_history=True)
    context = build_context(ip.history_manager)

    assert len(context.messages) == 1
    assert context.messages[0] == {"content": "a = 1", "role": "user"}


def test_build_context_start_stop(ip):
    # Test build_context with start and stop parameters
    ip.run_cell("a = 2", store_history=True)
    ip.run_cell("a + 1", store_history=True)
    ip.run_cell("a * 2", store_history=True)

    context = build_context(ip.history_manager, start=2, stop=3)

    assert len(context.messages) == 2
    assert context.messages[0] == {"content": "a + 1", "role": "user"}
    assert context.messages[1]["role"] == "system"
    assert context.messages[1]["content"].startswith("3")


def test_build_context_no_output(ip):
    # Test build_context with input and no output
    ip.run_cell("a = 1", store_history=True)
    context = build_context(ip.history_manager)

    assert len(context.messages) == 1
    assert context.messages[0] == {"content": "a = 1", "role": "user"}


def test_build_context_pandas_dataframe(ip):
    # Test build_context with pandas DataFrame
    ip.run_cell("import pandas as pd", store_history=True)
    ip.run_cell("df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})", store_history=True)
    ip.run_cell("df", store_history=True)
    context = build_context(ip.history_manager)

    assert len(context.messages) == 4
    assert context.messages[0] == {"content": "import pandas as pd", "role": "user"}
    assert context.messages[1] == {
        "content": "df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})",
        "role": "user",
    }
    assert context.messages[2] == {"content": "df", "role": "user"}

    markdown_repr = context.messages[3]["content"]
    assert context.messages[3]["role"] == "system"

    # We're sampling a two by two so we can just check the permutations

    if "|    |   A |   B |" in markdown_repr:
        assert "|    |   A |   B |" in markdown_repr
        assert "|---:|----:|----:|" in markdown_repr
        # The rows may be in another order so we just check containment
        assert "|  0 |   1 |   3 |" in markdown_repr
        assert "|  1 |   2 |   4 |" in markdown_repr
    else:
        assert "|    |   B |   A |" in markdown_repr
        assert "|---:|----:|----:|" in markdown_repr
        assert "|  0 |   3 |   1 |" in markdown_repr
        assert "|  1 |   4 |   2 |" in markdown_repr


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
