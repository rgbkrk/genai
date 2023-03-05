"""Magic to generate code cells for notebooks using OpenAI's API."""

from IPython import get_ipython
from IPython.core.magic import cell_magic
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
    help="Show additional information in the cell output",
)
@argument(
    "--in-place",
    action="store_true",
    help="Replace the current cell with the generated code",
)
@cell_magic
def assist(line, cell):
    """Generate code cells for notebooks using OpenAI's API.

    Usage:

    ```python
    %%assist
    # create a scatterplot from df
    ```

    Will create a code cell below the current one looking something like this

    ```python
    # create a scatterplot from df

    df.plot.scatter(x="col1", y="col2")
    ```

    If you've run previous cells in the notebook, the generated code will be
    based on the history of the current notebook session.

    That even means that, for example, running `df.columns` in a previous cell
    will cause generated code to use column names in the future!

    Caveats:

    - Only the last 5 cell executions are provided as context.
    - The generated code is not guaranteed to be correct, idiomatic, efficient, readable, or useful.
    - The generated code is not guaranteed to be syntactically correct or even something to write home about.

    There are several options you can use to control the behavior of the magic.

    --fresh: Ignore the history of the current notebook session and generate code from scratch.

    --in-place: Replace the current cell with the generated code.

    --verbose: Show additional information in the cell output.

    Example:

    ```python
    %%assist --fresh --in-place --verbose
    # how can I query for pokemon via the PokéAPI?
    ```
    """
    progress = display(HTML(starting_message()), display_id=True)

    args = parse_argstring(assist, line)

    ip = get_ipython()
    cell_text = cell.strip()

    context = []
    if not args.fresh:
        context = get_historical_context(ip)

    if args.verbose:
        print("magic arguments:", line)
        print("submission:", cell)
        print("context:", context)

    generated_text = generate_next_cell(context, cell_text)

    progress.update(HTML(completion_made()))

    new_cell = f"""{cell_text}\n{generated_text}"""

    if args.in_place:
        new_cell = f"""#%%assist {line}\n{cell_text}\n{generated_text}"""

    ip.set_next_input(
        new_cell,
        replace=args.in_place,
    )
