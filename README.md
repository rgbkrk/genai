[Install](#installation) | [License](./LICENSE) | [Code of Conduct](./CODE_OF_CONDUCT.md) | [Contributing](./CONTRIBUTING.md)

# GenAI: generative AI tooling for IPython

Generate code cells and get recommendations after exceptions in all Jupyter environments, including IPython, JupyterLab, Jupyter Notebook, and Noteable.

TL;DR

```
%pip install genai
%load_ext genai
```

## Genai In Action

![Genai making a suggestion followed by running suggested code](https://user-images.githubusercontent.com/836375/225177905-17cfb526-60f8-486d-b468-60a6a01db02e.gif)

- [Blog Post](https://noteable.io/blog/introducing-genai/)
- [Example Notebook](https://app.noteable.io/f/1605d16d-f5d3-4099-8fec-2ca727075b3b/Introducing-Genai.ipynb)

<!-- --8<-- [start:intro] -->

## Introduction

We've taken the context from IPython, mixed it with OpenAI's Large Language Models, and given you the power to generate code cells and get recommendations after exceptions in all Jupyter environments, including IPython, JupyterLab, Jupyter Notebook, and Noteable.

<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:requirements] -->

## Requirements

Python 3.8+

<!-- --8<-- [end:requirements] -->

<!-- --8<-- [start:install] -->

## Installation

### Poetry

```shell
poetry add genai
```

### Pip

```shell
pip install genai
```

<!-- --8<-- [end:install] -->

<!-- --8<-- [start:start] -->

## Loading the IPython extension

Make sure to set the `OPENAI_API_KEY` environment variable first before using it in IPython or your [preferred notebook platform of choice](https://noteable.io/).

```
%load_ext genai
```

## Features

- `%%assist` magic command to generate code from natural language
- Custom exception suggestions

### Custom Exception Suggestions

```python
In [1]: %load_ext genai

In [2]: import pandas as pd

In [3]: df = pd.DataFrame(dict(col1=['a', 'b', 'c']), index=['first', 'second', 'third'])

In [4]: df.sort_values()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[4], line 1
----> 1 df.sort_values()

File ~/.pyenv/versions/3.9.9/lib/python3.9/site-packages/pandas/util/_decorators.py:331, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
    325 if len(args) > num_allow_args:
    326     warnings.warn(
    327         msg.format(arguments=_format_argument_list(allow_args)),
    328         FutureWarning,
    329         stacklevel=find_stack_level(),
    330     )
--> 331 return func(*args, **kwargs)

TypeError: sort_values() missing 1 required positional argument: 'by'
```

#### 💡 Suggestion

The error message is indicating that the `sort_values()` method of a pandas dataframe is missing a required positional argument.

The `sort_values()` method requires you to pass a column name or list of column names as the `by` argument. This is used to determine how the sorting will be performed.

Here's an example:

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Carol', 'David', 'Eva'],
    'Age': [32, 24, 28, 35, 29],
    'Salary': [60000, 40000, 35000, 80000, 45000]
})

# sort by Age column:
df_sorted = df.sort_values(by='Age')
print(df_sorted)
```

In this example, the `by` argument is set to `'Age'`, which sorts the dataframe by age in ascending order. Note that you can also pass a list of column names if you want to sort by multiple columns.

## Example

```python
In [1]: %load_ext genai

In [2]: %%assist
   ...:
   ...: # Pull census data
   ...:
'What would a data analyst do? 🤔'

In [3]: # generated with %%assist
   ...: # Pull census data
   ...: # To pull census data we can use the `requests` library to send a GET request to the appropriate API endpoint.
   ...: # First, import the requests module
   ...: import requests
   ...:
   ...: # Define the URL endpoint to the Census API
   ...: url = "https://api.census.gov/data/2019/pep/population"
   ...:
   ...: # Define the parameters needed for the API request, such as dataset and variables requested
   ...: params = {
   ...:     "get": "POP",
   ...:     "for": "state:*",
   ...: }
   ...:
   ...: # Send a GET request to the Census API endpoint with the parameters
   ...: response = requests.get(url, params=params)
   ...:
   ...: # Access the response content
   ...: content = response.content
   ...:
   ...: # The Census data is now stored in the `content` variable and can be processed or saved elsewhere. The user can modify the `params` variable to request different data or specify a different API endpoint.

In [6]: content
Out[6]: b'[["POP","state"],\n["4903185","01"],\n["731545","02"],\n["7278717","04"],\n["3017804","05"],\n["39512223","06"],\n["5758736","08"],\n["973764","10"],\n["705749","11"],\n["3565287","09"],\n["21477737","12"],\n["10617423","13"],\n["1787065","16"],\n["1415872","15"],\n["12671821","17"],\n["6732219","18"],\n["3155070","19"],\n["2913314","20"],\n["4467673","21"],\n["4648794","22"],\n["1344212","23"],\n["6045680","24"],\n["6892503","25"],\n["9986857","26"],\n["5639632","27"],\n["2976149","28"],\n["6137428","29"],\n["1068778","30"],\n["1934408","31"],\n["3080156","32"],\n["1359711","33"],\n["8882190","34"],\n["2096829","35"],\n["19453561","36"],\n["10488084","37"],\n["762062","38"],\n["11689100","39"],\n["3956971","40"],\n["4217737","41"],\n["12801989","42"],\n["1059361","44"],\n["5148714","45"],\n["884659","46"],\n["6829174","47"],\n["28995881","48"],\n["623989","50"],\n["3205958","49"],\n["8535519","51"],\n["7614893","53"],\n["1792147","54"],\n["5822434","55"],\n["578759","56"],\n["3193694","72"]]'
```

<!-- --8<-- [end:start] -->
