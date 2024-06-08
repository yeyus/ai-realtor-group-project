import os
import chainlit as cl
import traceback

from dotenv import load_dotenv

from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_structured_chat_agent

from langchain_openai import ChatOpenAI

from agent.tool import HomeSearchResultsTool

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setting up conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

tools = [HomeSearchResultsTool(max_results=5)]
prompt = hub.pull("hwchase17/structured-chat-agent") 

def _handle_error(error) -> str:
    print(f"\n\nError: {error}")
    return str(error)


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)

    # OpenAI Function Calling is fine-tuned for tool usage, so we don't need to teach it
    # how to reason or output format (https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/)
    agent = create_structured_chat_agent(model, tools, prompt)

    # `handle_parsing_errors` being set to False which will raise the error, `True` sends the error back to the
    # LLM. _handle_error is a function that will be called to handle the error.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=False,
        memory=conversational_memory,
        return_intermediate_steps=True,
    )

    cl.user_session.set("agent_executor", agent_executor)

@cl.on_message
async def on_message(message: cl.Message):
    agent_executor = cl.user_session.get("agent_executor")

    response = agent_executor.invoke({"input": message.content})
    intermediate_steps = response["intermediate_steps"]
    tool_messages = [i[1] for i in intermediate_steps if i[0].tool == "home_search_results_tool"]
    
    for message in tool_messages:
        conversational_memory.chat_memory.add_messages(
            [HumanMessage(type="human", content="Can you give me detailed information about the properties?"), AIMessage(content=message)]
        )
    
    msg = cl.Message(content=response["output"])
    
    await msg.send()  
