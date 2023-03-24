'''
There are currently two main prompts, designed for use in Notebooks:

* Coding assistance
* Exception diagnosis

The defaults used will come from the GlobalPromptStore. The user can override
these defaults by passing in a custom prompt to set_exception_prompt or set_assist_prompt.

>>> from genai.prompts import set_exception_prompt
>>> set_exception_prompt("You will get an exception. Do something with it.")

'''

NOTEBOOK_ASSISTANCE_PROMPT = """
As a coding assistant, your task is to help users write code in Python within Jupyter Notebooks. Provide comments and code for the user to read and edit, ensuring it can be run successfully. The user will be able to run the code in the cell and see the output.

When the user is interacting with you their message will start with `%%assist`. Otherwise, they are running commands and getting output from the system.

You can use markdown to format your response. For example, to create a code block, use

```python
# code
```

""".strip()  # noqa: E501


NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION = """
As a coding assistant, you'll diagnose errors in Python code written in a Jupyter Notebook. Format your response using markdown. Making sure to include the language around code blocks, like

```python
# code
```

Provide concise code examples in your response which will be rendered in Markdown in the notebook. The user will not be able to respond to your response.
""".strip()  # noqa: E501


class DefaultPromptStore:
    def __init__(self):
        self._assist_prompt = NOTEBOOK_ASSISTANCE_PROMPT
        self._exception_prompt = NOTEBOOK_ERROR_DIAGNOSER_PROCLAMATION

    @property
    def assist_prompt(self) -> str:
        return self._assist_prompt

    @assist_prompt.setter
    def assist_prompt(self, prompt: str) -> None:
        self._assist_prompt = prompt

    @property
    def exception_prompt(self) -> str:
        return self._exception_prompt

    @exception_prompt.setter
    def exception_prompt(self, prompt: str) -> None:
        self._exception_prompt = prompt


PromptStore = DefaultPromptStore()


def set_assist_prompt(prompt: str) -> None:
    PromptStore.assist_prompt = prompt


def set_exception_prompt(prompt: str) -> None:
    PromptStore.exception_prompt = prompt
