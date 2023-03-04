# GenAI: generative AI tooling for IPython

## Installation

```
%pip install genai
```

## Loading the IPython extension

Make sure to set the `OPENAI_API_KEY` environment variable first before using it in IPython or your [preferred notebook platform of choice](https://noteable.io/).

```
%load_ext genai
```

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
