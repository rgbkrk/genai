import os
import random
from genai.components import collapsible_log, completion_viewer, field, styled_code

import openai

from vdom import pre, div, h3, b as bold

from IPython.core.magic import (
    register_cell_magic,
)
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from IPython.display import display

"""
Initialize the openai API.
"""
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise Exception("Please set the OPENAI_API_KEY environment variable")

"""
If the users puts #ignore or #keep at the top of their cells, they can control
what gets sent in a prompt to openai.
"""
# Cells we want to ignore
ignore_tokens = [
    "# genai:ignore",
    "#ignore",
    "# ignore",
    "%%assist",
    "get_ipython",
    "%load_ext",
    "import genai",
    "%pip install",
    "#%%assist",
]

# Cells we want to keep for sure, in case In[] is too long.
keep_tokens = ["#keep", "# keep"]


def ignored(inp):
    for ignore in ignore_tokens:
        if inp.startswith(ignore):
            return True
    return False


def estimate_tokens(cells: list[str]):
    """
    Estimate the number of tokens in a cell.
    """
    return len("".join(cells).split())


def truncate_prior_cells(cells: list[str], max_tokens: int = 500):
    """
    Truncate the priors to a reasonable length.
    """
    if max_tokens < 0:
        return []

    # Split into halves, with a preference for using #keep cells.
    first_half = cells[: len(cells) // 2]
    second_half = cells[len(cells) // 2 :]

    keepers = []

    # Only include the keeps from the first half
    for cell in first_half:
        for keep in keep_tokens:
            if keep in cell:
                keepers.append(cell)
                break

    if estimate_tokens(keepers) > max_tokens:
        # TODO: Trim down the keeps as an option
        raise Exception("Too many cells are marked as #keep")

    # Check if keeps + second_half is the right size
    if estimate_tokens(keepers + second_half) < max_tokens:
        return keepers + second_half

    # Split the second half now to see if we can fit more in.
    trimmed_second = truncate_prior_cells(
        second_half, max_tokens - estimate_tokens(keepers)
    )

    if estimate_tokens(keepers + trimmed_second) < max_tokens:
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

    if estimate_tokens(lines) > max_length:
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
                "Just a little bit of data engineering will fix this 🔧",
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


@magic_arguments()
@argument("--fresh", action="store_true")
@argument(
    "-t",
    "--temperature",
    type=float,
    default=0.5,
    help="What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.",
)
@argument(
    "--verbose",
    action="store_true",
    help="Show a full log in the cell output",
)
@argument(
    "--in-place",
    action="store_true",
    help="Replace the current cell with the generated code",
)
@register_cell_magic
def assist(line, cell):
    progress = display(starting_message(), display_id=True)

    args = parse_argstring(assist, line)
    ip = get_ipython()

    # noop
    def log(*args, **kwargs):
        pass

    if args.verbose:
        handle = display(
            collapsible_log(),
            display_id=True,
        )
        logs = []

        def log(element):
            # Add VDOM element
            logs.append(element)
            handle.update(collapsible_log(logs))

        log(
            div(
                h3("magic arguments"),
                styled_code(line),
                h3("submission"),
                styled_code(cell),
            )
        )

    previous_inputs = ""
    if not args.fresh:
        previous_inputs = prior_code(ip.history_manager.input_hist_raw).strip()

        if len(previous_inputs) > 0:
            log(div(bold("Previous inputs"), styled_code(previous_inputs)))
        else:
            log(
                div(
                    bold(
                        "No previous inputs to send in the prompt",
                        style={"color": "lightgrey"},
                    )
                )
            )

    cell_text = cell.strip()

    prompt = (
        f"""# Python code, matplotlib inline\n{previous_inputs}\n{cell_text}""".strip()
    )

    # make a rough estimate of the number of tokens being sent
    prompt_token_count = len(prompt.split()) * 2
    log(
        div(
            bold(f"Prompt being sent with an estimated {prompt_token_count} tokens:"),
            styled_code(prompt),
            style={"margin": "0.5em 0"},
        ),
    )

    #
    # Reference on tokens: https://beta.openai.com/tokenizer
    #
    # The token count of our prompt plus `max_tokens` cannot exceed the model's
    # context length, which is either 2048 or 4096.
    #
    # To assist us with this as a first pass, we'll calculate the recommended max
    # based on how many lines their first cell had. If it goes beyond, we'll

    # Scale the number of tokens according to the number of lines in the cell
    max_tokens = 50 * len(cell.split("\n"))

    # Remove the estimated prompt token count from the max
    if max_tokens + prompt_token_count > 4000:
        max_tokens = 4000 - prompt_token_count
    if max_tokens < 0:
        max_tokens = 256

    log(
        div(
            bold(f"Max tokens got set to {max_tokens}"),
            style={"margin": "0.5em 0"},
        ),
    )

    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=args.temperature,
    )

    progress.update(completion_made())

    log(completion_viewer(completion))

    choice = completion.choices[0]

    new_cell = f"""# generated with %%assist\n{cell_text}{choice.text}"""

    if args.in_place:
        new_cell = f"""#%%assist {line}\n{cell_text}{choice.text}"""

    ip.set_next_input(
        new_cell,
        replace=args.in_place,
    )
