from unittest.mock import patch

import pandas as pd
import pytest

from genai.context import (
    PastAssists,
    PastErrors,
    build_context,
    repr_genai,
    repr_genai_pandas,
    summarize_dataframe,
)
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


@pytest.mark.parametrize("patched_dataframe_sample", [1], indirect=True)
def test_build_context_pandas_dataframe(ip, patched_dataframe_sample):
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

    expected = """
## Dataframe Summary

Number of Rows: 2

Number of Columns: 2

### Column Information

|    | Column Name   | Data Type   |   Missing Values |   % Missing |
|----|---------------|-------------|------------------|-------------|
|  0 | A             | int64       |                0 |           0 |
|  1 | B             | int64       |                0 |           0 |

### Numerical Summary

|    | Column Name   |   count |   mean |      std |   min |   25% |   50% |   75% |   max |
|----|---------------|---------|--------|----------|-------|-------|-------|-------|-------|
|  0 | A             |       2 |    1.5 | 0.707107 |     1 |  1.25 |   1.5 |  1.75 |     2 |
|  1 | B             |       2 |    3.5 | 0.707107 |     3 |  3.25 |   3.5 |  3.75 |     4 |

### Categorical Summary

| Column Name   |
|---------------|

### Sample Data (2x2)

|    |   A |   B |
|----|-----|-----|
|  0 |   1 |   3 |
|  1 |   2 |   4 |

""".strip()
    assert markdown_repr == expected


@patch(
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
def test_build_context_assistance(create, ip):
    # Test build_context with assistance
    ip.run_cell("a = 1", store_history=True)
    ip.run_cell(
        f"""%%assist

Make the most dope plot ever    

""".strip(),
        store_history=True,
    )
    create.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "You should probably define b",
                },
            },
        ],
    }
    ip.run_cell("a * b", store_history=True)

    context = build_context(ip.history_manager)

    assert len(context.messages) == 6
    assert context.messages[0] == {"content": "a = 1", "role": "user"}
    assert context.messages[1] == {
        "content": "%%assist\n\nMake the most dope plot ever",
        "role": "user",
    }
    assert context.messages[2] == {
        "content": "superplot(df)",
        "role": "assistant",
    }
    assert context.messages[3] == {
        "content": "a * b",
        "role": "user",
    }

    assert context.messages[4]["role"] == "system"
    errorMessage = context.messages[4]["content"]
    assert "NameError: name 'b' is not defined" in errorMessage

    assert context.messages[5] == {
        "role": "assistant",
        "content": "## 💡 Suggestion\nYou should probably define b",
    }


def test_summarize_dataframe():
    # create a mock DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # call the summarize_dataframe function with the mock DataFrame
    summary = summarize_dataframe(df)

    # Check if the summary contains essential information
    assert "Number of Rows" in summary
    assert "Number of Columns" in summary
    assert "Column Information" in summary
    assert "Numerical Summary" in summary
    assert "Sample Data" in summary


@pytest.mark.parametrize("patched_dataframe_sample", [1], indirect=True)
def test_summarize_dataframe_no_missing(patched_dataframe_sample):
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    expected_output = """
## Dataframe Summary

Number of Rows: 3

Number of Columns: 2

### Column Information

|    | Column Name   | Data Type   |   Missing Values |   % Missing |
|----|---------------|-------------|------------------|-------------|
|  0 | A             | int64       |                0 |           0 |
|  1 | B             | int64       |                0 |           0 |

### Numerical Summary

|    | Column Name   |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|----|---------------|---------|--------|-------|-------|-------|-------|-------|-------|
|  0 | A             |       3 |      2 |     1 |     1 |   1.5 |     2 |   2.5 |     3 |
|  1 | B             |       3 |      5 |     1 |     4 |   4.5 |     5 |   5.5 |     6 |

### Categorical Summary

| Column Name   |
|---------------|

### Sample Data (3x2)

|    |   A |   B |
|----|-----|-----|
|  0 |   1 |   4 |
|  2 |   3 |   6 |
|  1 |   2 |   5 |

""".strip()

    actual = summarize_dataframe(df)

    assert actual == expected_output


@pytest.mark.parametrize("patched_series_sample", [1], indirect=True)
def test_repr_genai_pandas_series(patched_series_sample, ip):
    series = pd.Series([1, 2, 3])
    result = repr_genai_pandas(series)

    expected = """
## Series Summary

Number of Values: 3

Data Type: int64

Missing Values: 0 (0.00%)

### Summary Statistics

|    |   count |   mean |   std |   min |   25% |   50% |   75% |   max |
|----|---------|--------|-------|-------|-------|-------|-------|-------|
|  0 |       3 |      2 |     1 |     1 |   1.5 |     2 |   2.5 |     3 |

### Sample Data (3)

|    |   0 |
|----|-----|
|  0 |   1 |
|  2 |   3 |
|  1 |   2 |
""".strip()

    assert result == expected


@pytest.mark.parametrize("patched_series_sample", [1], indirect=True)
def test_repr_genai_pandas_series_pirate(patched_series_sample, ip):
    series = pd.Series(
        {
            'Name': 'Blackbeard',
            'Age': 40,
            'Ship': 'Queen Anne\'s Revenge',
            'Crew Size': 300,
            'Treasure': '$12.5 million',
        }
    )
    result = repr_genai_pandas(series)

    print(result)

    expected = """
## Series Summary

Number of Values: 5

Data Type: object

Missing Values: 0 (0.00%)

### Summary Statistics

|    |   count |   unique | top        |   freq |
|----|---------|----------|------------|--------|
|  0 |       5 |        5 | Blackbeard |      1 |

### Sample Data (5)

|           | 0                    |
|-----------|----------------------|
| Ship      | Queen Anne's Revenge |
| Age       | 40                   |
| Treasure  | $12.5 million        |
| Name      | Blackbeard           |
| Crew Size | 300                  |
""".strip()

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
