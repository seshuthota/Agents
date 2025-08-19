import config
from config import MODEL_NAME
from langchain.agents import create_react_agent, AgentExecutor, create_structured_chat_agent
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

search = DuckDuckGoSearchRun()
tools = [search]
prompt = PromptTemplate.from_template(
    "Answer the following questions as best you can. You have access to the following tools:\n\n{tools}\n\nUse the "
    "following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what "
    "to do\nAction: the action to take, should be one of [{tool_names}]\nAction Input: the input to the "
    "action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N "
    "times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input "
    "question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}")

llm = ChatOpenAI(model=MODEL_NAME)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

structured_agent = create_structured_chat_agent(llm=llm, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "What is a ReAct Agent and how does it work?"}))
