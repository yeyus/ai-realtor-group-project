import os
from pathlib import Path
import sys
import chainlit as cl
import traceback
import logging

from dotenv import load_dotenv

from langchain import hub
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_openai import ChatOpenAI

from agent.tool import HomeSearchResultsTool

# Configure logging
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(module)s - %(filename)s - %(levelname)s - %(message)s",
)

# Config
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY", None)
prompt_to_use = os.getenv("LANGCHAIN_PROMPT", "m3libea/ai-realtor")

# Setting up conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

tools = [HomeSearchResultsTool(max_results=5)]
prompt = hub.pull(prompt_to_use, api_key=langchain_api_key)


def _handle_error(error) -> str:
    print(f"\n\nError: {error}")
    return str(error)


@cl.on_chat_start
async def on_chat_start():
    logging.info("Chat started, using prompt %s", prompt_to_use)
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)

    # OpenAI Function Calling is fine-tuned for tool usage, so we don't need to teach it
    # how to reason or output format (https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/)
    agent = create_structured_chat_agent(
        model,
        tools,
        prompt,
    )

    # Callback handler
    stdouthandler = StdOutCallbackHandler()

    # `handle_parsing_errors` being set to False which will raise the error, `True` sends the error back to the
    # LLM. _handle_error is a function that will be called to handle the error.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=False,
        memory=conversational_memory,
        callbacks=[stdouthandler],
    )

    cl.user_session.set("agent_executor", agent_executor)


@cl.on_message
async def on_message(message: cl.Message):
    agent_executor = cl.user_session.get("agent_executor")

    try:
        response = agent_executor.invoke({"input": message.content})
        msg = cl.Message(content=response["output"])
        await msg.send()

    except Exception as ex:
        print("\n\nError Stacktrace: \n\n")
        traceback.print_stack()
        _handle_error(ex)
