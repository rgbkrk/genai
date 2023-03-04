import random
import time

from vdom import b as bold
from vdom import br, details, div, h3
from vdom import i as italics
from vdom import p, pre, span, summary


def styled_code(code):
    return pre(code, style={"backgroundColor": "#e7e7e7", "padding": "1em"})


# vdom component for visualizing OpenAI completion choices
def render_choices(entry):
    return div(
        [
            field("Index", entry["index"]),
            field("Finish Reason", entry["finish_reason"]),
            field("Logprobs", entry["logprobs"]),
            styled_code(entry["text"]),
        ]
    )


# Simple name/value pair that always coerces values to a string
def field(name, value):
    return div(bold(name), ": ", str(value))


# vdom component for visualizing OpenAI completion results
def completion_viewer(completion):

    # Render the completion results
    return div(
        field("Completion ID", completion["id"]),
        field("Model", completion["model"]),
        field("Object", completion["object"]),
        field(
            "Created At",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completion["created"])),
        ),
        div(
            [field(usage, completion["usage"][usage]) for usage in completion["usage"]]
        ),
        h3("Choices", style={"marginBottom": ".5em"}),
        field("# of Choices", len(completion["choices"])),
        br(),
        div([render_choices(entry) for entry in completion["choices"]]),
    )


def collapsible_log(children=None, title=None):

    if children is None or len(children) == 0:
        children = div("No logs yet.")

    if title is None:
        title = [bold("Debug Log"), " - ", italics("click to expand/collapse")]

    return details(summary(title), div(children))


def starting_message():
    return random.choice(
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


def completion_made():
    return random.choice(
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
