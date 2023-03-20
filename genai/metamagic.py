import openai
from IPython import get_ipython

from genai.display import GenaiMarkdown
from genai.generate import deltas

# Creating a setup that will make it easy to define new magics
# that rely on a prompt and a response. This will be useful for
# defining a magic that will allow the user to define a prompt
# and then generate a response to that prompt.


def create_gpt_magic(magic_name, prompt, generate_context=None, model="gpt-3.5-turbo"):
    # Define a cell magic that uses `magic_name`

    def magic(line, cell):
        gm = GenaiMarkdown(execution_count=get_ipython().execution_count)
        gm.display()

        context = []
        if generate_context:
            context = generate_context(line=line, cell=cell)
        else:
            context = [
                {
                    "role": "user",
                    "content": cell,
                },
            ]

        # Create the response from ChatCompletion
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                # Establish the context of the conversation
                {
                    "role": "system",
                    "content": prompt,
                },
                *context,
            ],
            stream=True,
        )

        gm.consume(deltas(response))

    get_ipython().register_magic_function(magic, magic_kind="cell", magic_name=magic_name)
