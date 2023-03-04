import openai

NOTEBOOK_CODING_ASSISTANT_TEMPLATE = """You are a notebook coding assistant,
designed to help users write their next code cell. The primary language is python.
Write comments and code for the user to read and run. The user will be able to
edit the code and run it.
"""


def generate_next_cell(
    context,  # List[Dict[str, str]]
    text,
):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # Establish the context in which GPT will respond
            {
                "role": "system",
                "content": NOTEBOOK_CODING_ASSISTANT_TEMPLATE,
            },
            # In, Out
            *context,
            # The user code/text
            {
                "role": "user",
                "content": text,
            },
        ],
    )

    text = completion["choices"][0]["message"]["content"]

    return text
