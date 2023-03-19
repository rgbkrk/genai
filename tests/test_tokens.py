from genai.tokens import MAX_TOKENS, num_tokens_from_messages, trim_messages_to_fit_token_limit


def test_num_tokens_from_messages():
    messages = [
        {"role": "system", "content": "You are a leet coder.", "name": "GenAI"},
        {"role": "user", "content": "2+2"},
        {
            "role": "assistant",
            "content": "#wow\n\n```python\nprint(2+2)\n```\n\n```python\n4\n```",
        },
    ]
    num_tokens = num_tokens_from_messages(messages)

    assert num_tokens == 48


def test_num_tokens_from_messages_gpt_4():
    messages = [
        {
            "role": "system",
            "content": "You are a leet coder. like seriously just amazing, so cool omg",
        },
        {"role": "user", "content": "2+2"},
        {
            "role": "assistant",
            "content": "#whoa\n\n```python\n2+2\n```\n\n```python\n4\n```",
        },
    ]
    num_tokens = num_tokens_from_messages(messages, model="gpt-4-0314")

    assert num_tokens == 55


def test_trim_messages_to_fit_token_limit():
    messages = [
        {"role": "system", "content": "You are a leet coder."},
        {"role": "system", "content": "Like seriously just amazing, so cool omg"},
        {"role": "user", "content": "2+2", "name": "calculator"},
        {
            "role": "assistant",
            "content": "#wow\n\n```python\nprint(2+2)\n```\n\n```python\n4\n```",
        },
    ]
    model = "gpt-3.5-turbo-0301"
    trimmed_messages = trim_messages_to_fit_token_limit(messages, model=model, max_tokens=32)
    expected_messages = [
        {
            "role": "assistant",
            "content": "#wow\n\n```python\nprint(2+2)\n```\n\n```python\n4\n```",
        },
    ]

    assert trimmed_messages == expected_messages
