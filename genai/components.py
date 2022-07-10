import time

from vdom import h3, div, pre, b as bold, i as italics, p, details, summary, span, br


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
