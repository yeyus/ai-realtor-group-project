---
title: Proper Bot
emoji: 🏡
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
---

# Real Estate - Group assignment

### Datasets

- Historical Data: [Historical Housing Data](https://www.car.org/en/marketdata/data/housingdata)
- Zillow Housing Data: [Housing Data - Zillow Research](https://www.zillow.com/research/data/)
- California Data 1990: [California Housing Prices | Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

### Examples

- [House Price Prediction 🏡 Beginner's Notebook | Kaggle](https://www.kaggle.com/code/heyrobin/house-price-prediction-beginner-s-notebook)
- [House Prices Prediction using TFDF](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf) : This uses Tensor Flow with Decision Forest. More info about [Decision Forest ](https://www.tensorflow.org/decision_forests)
- [House Prediction](https://huggingface.co/spaces/rsatish1110/HousePricePrediction/blob/main/app.py) on Hugging Face, it is currently broken because of an issue with hw quota but it references a [dataset](https://github.com/ageron/handson-ml2/tree/master/datasets/housing) that we can take as reference too. It uses radio, which is a library Hugging face provides for easy UI for ML models.
- [Price Recommender](https://huggingface.co/spaces/yxmauw/ames-houseprice-recommender/blob/main/app.py)

### What should the model answer?

This section is to talk about the different options we have to give answers to the users:

- Predict the right price for your house (sellers)
- Predict the value of houses for new home buyers.
- Ask for the requirements to the buyers and give options about the areas they should look for.

### Design

- Which Dataset to use?
- Chatbot? prediction based on data?

#### Dataset

## Agent

### Description

Used Agent is: `"hwchase17/structured-chat-agent"`. The structured chat agent is a simple chatbot that can answer questions about structured data. The agent is using the OpenAI API to generate the response. It is currently using the `gpt-3.5-turbo-0125` model, which is the cheapest model available in the OpenAI API.

### How to run the code locally

Configure the Python environment with the following environment variables and requirements:

1. Create `.env` file with the following content:

```
OPENAI_API_KEY=<YOUR_API_KEY>
```

2. Install the requirements

```
pip install -r requirements.txt
```

### Run the code

```
chainlit run app.py
```

## Libraries

- https://github.com/Bunsly/HomeHarvest
- https://pypi.org/project/langchain-openai/
- https://chainlit.io/
