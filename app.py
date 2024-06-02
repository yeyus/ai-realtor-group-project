import os
import chainlit as cl
from dotenv import load_dotenv

from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.agents import AgentExecutor, create_structured_chat_agent

from langchain_openai import ChatOpenAI

from agent.tool import HomeSearchResultsTool

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# # Setting up conversational memory
# conversational_memory = ConversationBufferWindowMemory(
#     memory_key='chat_history',
#     k=5,
#     return_messages=True
# )

tools = [HomeSearchResultsTool(max_results=5)]
# prompt = prompt = hub.pull("hwchase17/react-chat") 
prompt = hub.pull("hwchase17/structured-chat-agent") 

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)    
    agent = create_structured_chat_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    cl.user_session.set("agent_executor", agent_executor)


@cl.on_message
async def on_message(message: cl.Message):
    agent_executor = cl.user_session.get("agent_executor")
    
    response = agent_executor.invoke({"input": message.content})
    msg = cl.Message(content=response["output"])

    await msg.send()
