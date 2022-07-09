import os
import random

import openai

from vdom import pre
from IPython.core.magic import (
    register_cell_magic,
)

from IPython.display import display

"""
Initialize the openai API.
"""
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise Exception("Please set the OPENAI_API_KEY environment variable")

if openai.organization is None:
    raise Exception("Please set the OPENAI_ORGANIZATION environment variable")

"""
If the users puts #ignore or #keep at the top of their cells, they can control
what gets sent in a prompt to openai.
"""
# Cells we want to ignore
ignore_tokens = [
    "#ignore",
    "# ignore",
    "%%assist",
    "get_ipython",
    "%load_ext",
    "import genai",
    "%pip install",
]

# Cells we want to keep for sure, in case In[] is too long.
keep_tokens = ["#keep", "# keep"]


def ignored(inp):
    for ignore in ignore_tokens:
        if inp.startswith(ignore):
            return True
    return False


def truncate_prior_cells(cells: list[str], max_length: int = 500):
    """
    Truncate the priors to a reasonable length.
    """
    if max_length < 0:
        return []

    # Split into halves, with a preference for using #keep cells.
    first_half = cells[: len(cells) // 2]
    second_half = cells[len(cells) // 2 :]

    keepers = []

    # Only include the keeps from the first half
    for keep in keep_tokens:
        if first_half.startswith(keep):
            keepers.append(keep)

    if len("".join(keepers)) > max_length:
        # TODO: Trim down the keeps as an option
        raise Exception("Too many cells are marked as #keep")

    # Check if keeps + second_half is the right size
    if len("".join(keepers)) + len("".join(second_half)) < max_length:
        return keepers + second_half

    # Split the second half now to see if we can fit more in.
    trimmed_second = truncate_prior_cells(
        second_half, max_length - len("".join(keepers))
    )

    if len("".join(keepers)) + len("".join(trimmed_second)) < max_length:
        return keepers + trimmed_second

    return keepers


def prior_code(inputs: list[str], max_length: int = 500):
    """
    Grab all of the inputs, up to a certain length, and return them as a string.
    """

    # Add lines in reverse order
    lines = []

    # Assume, naively at first that we can keep it all.
    for inp in inputs:
        if ignored(inp):
            continue
        lines.append(inp)

    if len(lines) > max_length:
        lines = truncate_prior_cells(lines, max_length)

    return "".join([inp.replace("# generated with %%assist", "") for inp in lines])


"""
Messages were generated using the text-davinci-002 model using this prompt:

Write some catch phrases for an AI that generates code cells for Jupyter
notebooks. The AI was trained on the work of other data scientists, analysts,
data engineers, programmers, and quirky artists. Crack some jokes, have some
fun. Be a quirky AI.

[
    "Phoning a friend 📲",
"""


def starting_message():
    return pre(
        random.choice(
            [
                "Phoning a friend 📲",
                "Reaching out to another data scientist 📊",
                "Just a little bit of data engineering 🔧",
                "Trying my best 💯",
                "Generating some code cells 💻",
                "Asking the internet 🌐",
                "Searching through my memory 💾",
                "What would a data analyst do? 🤔",
                "Querying my database 🗃️",
                "Running some tests 🏃‍",
                "One code cell, coming right up! 🚀",
                "I'm a machine, but I still enjoy helping you code. 😊",
            ]
        )
    )


def completion_made():
    return pre(
        random.choice(
            [
                "Enjoy your BRAND NEW CELL 🚙",
                "Just what you needed - more code cells! 🙌",
                "Here's to helping you code! 💻",
                "Ready, set, code! 🏁",
                "Coding, coding, coding... 🎵",
                "Just another code cell... 🙄",
                "Here's a code cell to help you with your analysis! 📊",
                "Need a code cell for your data engineering work? I got you covered! 🔥",
                "And now for something completely different - a code cell! 😜",
                "I got a little creative with this one - hope you like it! 🎨",
                "This one's for all the data nerds out there! 💙",
            ]
        )
    )


@register_cell_magic
def assist(line, cell):
    ip = get_ipython()

    previous_inputs = prior_code(ip.history_manager.input_hist_raw)
    cell_text = "".join(cell)

    prompt = f"""# Python\n{previous_inputs}\n{cell_text}""".strip()

    progress = display(starting_message(), display_id=True)

    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=len(prompt) + 400,
        temperature=0.6,
    )
    progress.update(completion_made())

    choice = completion.choices[0]
    ip.set_next_input(
        f"""# generated with %%assist\n{cell_text}{choice.text}""", replace=False
    )
