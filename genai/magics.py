"""Magic to generate code cells for notebooks using OpenAI's API."""

from IPython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.display import display, HTML

from genai.components import completion_made, starting_message
from genai.context import get_historical_context
from genai.generate import generate_next_cell


@magic_arguments()
@argument("--fresh", action="store_true")
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
    progress = display(HTML(starting_message()), display_id=True)

    args = parse_argstring(assist, line)

    ip = get_ipython()
    cell_text = cell.strip()

    context = []
    if not args.fresh:
        context = get_historical_context(ip)

    if args.verbose:
        display(
            f"""magic arguments: {line}
submission: {cell}"""
        )

    generated_text = generate_next_cell(context, cell_text)

    progress.update(HTML(completion_made()))

    new_cell = f"""{cell_text}\n{generated_text}"""

    if args.in_place:
        new_cell = f"""#%%assist {line}\n{cell_text}\n{generated_text}"""

    ip.set_next_input(
        new_cell,
        replace=args.in_place,
    )
