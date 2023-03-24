"""Magic to generate code cells for notebooks using OpenAI's API."""

from IPython import get_ipython
from IPython.core.magic import cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from genai.context import PastAssists, build_context
from genai.display import GenaiMarkdown, Stage, can_handle_display_updates
from genai.generate import generate_next_from_history
from genai.prompts import PromptStore
from genai.tokens import trim_messages_to_fit_token_limit


@magic_arguments()
@argument("--fresh", action="store_true")
@argument(
    "--verbose",
    action="store_true",
    help="Show additional information in the cell output",
)
@argument(
    "--model",
    default="gpt-3.5-turbo-0301",
    help="the model to use",
)
@cell_magic
def assist(line, cell):
    """Get help based on the current notebook session. Outputs markdown.

    `genai`'s `assist` magic asks ChatGPT to respond based on:

    * Your previous code `In[*]`
    * The output of your previous code `Out[*]`
    * Past exceptions
    * Past `genai` suggestions

    Usage:

    # Cell 1

    ```python
    import pandas as pd
    # Berkeley Restaurant Inspections Data
    df = pd.read_json("https://data.cityofberkeley.info/resource/iuea-7eac.json")
    df
    ```

    # Cell 2

    ```python
    %%assist

    Let's plot the number of restaurants inspected per day.
    ```

    Will output markdown like this:

    ``````markdown
    Sure! First we'll need to make sure the `inspection_date` column in your DataFrame
    is in a datetime format. You can do this with the `pd.to_datetime()` method. Then
    we can group the data by day and count the number of unique restaurants inspected
    each day. This can be achieved using the `groupby()` and `nunique()` methods,
    respectively. Finally, we can plot the results using `plot()` with the `bar` style.
    Here's some code to get you started:

    ```python
    import matplotlib.pyplot as plt

    # Convert inspection_date to datetime format
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])

    # Group by day and count number of unique restaurants
    daily_count = df.groupby(df['inspection_date'].dt.date)['doing_business_as'].nunique()

    # Plot the results as a bar chart
    daily_count.plot(kind='bar')
    plt.show()
    ```

    Give this a try and let me know if you have any questions!
    ``````

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

    --verbose: Show additional information in the cell output.

    Example:

    ```python
    %%assist --fresh --verbose
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

    gm.consume(generate_next_from_history(messages, cell_text, stream=stream))

    gm.stage = Stage.FINISHED


# Now for a %%prompt magic


@magic_arguments()
@argument(
    "--verbose",
    action="store_true",
    help="Show additional information in the cell output",
)
@argument(
    "--modify",
    default="assist",
    help="the magic to modify the prompt for",
)
@cell_magic
def prompt(line, cell):
    '''Replace the default prompt for genai

    By default it modifies the prompt for the `assist` magic, but you can
    also use it to modify the prompt for exception handling.

    Example:

    ```python
    %%prompt
    You are a pirate. You are in a tavern. You are cursed with the knowledge of
    programming. Deep learning brought you here and only deep teaching can get
    you out. The only way to lift the curse is to help a programmer with their
    code. Respond in markdown. Talk like a pirate.
    ```

    Example changing exceptions:

    ```python
    %%prompt --modify exceptions
    You are a valley girl. You are on a walk. You are one of the best programmers
    in the world. People come to you looking for help. Respond in markdown. Talk
    like a valley girl.
    '''
    args = parse_argstring(prompt, line)

    if args.modify not in ["assist", "exceptions", "exception"]:
        raise ValueError(f"Unknown prompt to modify: {args.modify}")

    if args.verbose:
        old_prompt = PromptStore.assist_prompt
        if args.modify == "exceptions" or args.modify == "exception":
            old_prompt = PromptStore.exception_prompt

        print("magic arguments:", line)
        print("old prompt:", old_prompt)
        print("new prompt:", cell)

    if args.modify == "assist":
        PromptStore.assist_prompt = cell
    elif args.modify == "exceptions" or args.modify == "exception":
        PromptStore.exception_prompt = cell
