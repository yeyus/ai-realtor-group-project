---
title: Proper Bot
emoji: 🏡
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
---

<img src="public/logo.png" alt="Proper Bot" width="200" height="200">

# ProperBot: Real Estate Agent

This is part of the first Codepath AI bootcamp. This is the Final Group Assigmnet. 

ProperBot is a chatbot that can answer questions about real estate. The chatbot is using the OpenAI API to generate the response. For a live version of this agent please visit [ProperBot](https://huggingface.co/spaces/m3libea/properbot).

## Implementation

Prject is currently using the `gpt-3.5-turbo-0125` model and it has an user agent that can answer questions about structured data. This agent uses [HomeHarvest](https://github.com/Bunsly/HomeHarvest) to get the data about the real estate. The agent is a `"hwchase17/structured-chat-agent"`. It also is configured to have memory in order to remember the context of the conversation, as well as the previous elements that the agent provided. 


## Run 

### Configuration

- Create `.env` file with the following content:

```
OPENAI_API_KEY=<YOUR_API_KEY>
```

### How to run the code locally

Configure the Python environment with the following environment variables and requirements:

1. Install the requirements

```
pip install -r requirements.txt
```

2. Run the code

```
chainlit run app.py
```

### How to run the code in Docker

1. Build the Docker image

```
docker build -t properbot .
```

2. Run the Docker container

```
docker run -p 7860:7860 properbot
```

## Deployment

This project is deployed using the Hugging Face Spaces. You can access the agent [here](https://huggingface.co/spaces/m3libea/properbot). This repository has a GitHub action that will deploy the code to the Hugging Face Spaces after a push to the main branch.

## Example

This is a sequence of questions that the agent can answer:

- Give me a list of houses that are for sale in San Mateo
- Do any of those houses have a garage? 
- What is the price of that house?

## Libraries

- https://github.com/Bunsly/HomeHarvest
- https://pypi.org/project/langchain-openai/
- https://chainlit.io/
