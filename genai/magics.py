"""Magic to generate code cells for notebooks using OpenAI's API."""

from IPython import get_ipython
from IPython.core.magic import cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from genai.context import PastAssists, build_context
from genai.display import GenaiMarkdown, Stage, can_handle_display_updates
from genai.generate import generate_next_cell
from genai.tokens import trim_messages_to_fit_token_limit


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
@argument(
    "--model",
    default="gpt-3.5-turbo-0301",
    help="the model to use",
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
    ip = get_ipython()

    args = parse_argstring(assist, line)

    gm = GenaiMarkdown(
        stage=Stage.STARTING,
    )
    gm.display()
    PastAssists.add(ip.execution_count, gm)

    cell_text = cell.strip()

    model = args.model

    stream = can_handle_display_updates()

    messages = []
    if not args.fresh:
        # Start at 5 before the current execution to get the last 5 executions
        start = max(ip.execution_count - 5, 1)
        # Do not include the current execution
        stop = ip.execution_count

        context = build_context(ip.history_manager, start=start, stop=stop)
        messages = trim_messages_to_fit_token_limit(context.messages, model=model)

    if args.verbose:
        print("magic arguments:", line)
        print("submission:", cell)
        print("messages:", messages)

    gm.consume(generate_next_cell(messages, cell_text, stream=stream))

    gm.stage = Stage.FINISHED
