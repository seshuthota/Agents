import operator
import os
import warnings
from typing import Annotated, Tuple, List, TypedDict

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.chains.structured_output import create_openai_fn_runnable
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_agent_executor

warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LS_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Agents"

# Define tools
search = TavilySearchResults(max_results=1)
tools = [search]

# Define Execution Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo")

big_llm = ChatOpenAI(model="gpt-4-turbo-preview")

agent_runnable = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = create_agent_executor(agent_runnable=agent_runnable, tools=tools)


# Define State
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


# Planning Step
class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(default=[], description="Different steps to follow, should be in the sorted order.")


planner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. This plan should involve individual tasks, 
    that if executed correctly will yield the correct answer. Do not add any superfluous steps. The result of the 
    final step should be the final answer. Make sure that each step has all the information needed - do not skip 
    steps.
    
    Objective: {objective}
    """
)

planner = create_structured_output_runnable(
    Plan, big_llm, planner_prompt
)


# Re-Plan Step
class Response(BaseModel):
    """Response to the user"""
    response: str = Field(default="", title="Final answer to be given to the user based on the input question")


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = create_openai_fn_runnable(
    [Plan, Response],
    big_llm,
    replanner_prompt
)


# Create Graph

def execute_step(state: PlanExecute):
    task = state["plan"][0]
    agent_response = agent_executor.invoke({"input": task, "chat_history": []})
    return {
        "past_steps": (task, agent_response["agent_outcome"].return_values["output"])
    }


def plan_step(state: PlanExecute):
    plan = planner.invoke({"objective": state["input"]})
    return {"plan": plan.steps}


def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    if isinstance(output, Response):
        return {"response": output.response}
    else:
        return {"plan": output.steps}


def should_end(state: PlanExecute):
    if state["response"]:
        return True
    else:
        return False


workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)
workflow.add_node("replanner", replan_step)
workflow.add_node("agent", execute_step)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replanner")
workflow.add_conditional_edges(
    "replanner",
    should_end,
    {
        True: END,
        False: "agent"
    }
)

app = workflow.compile()

config = {"recursion_limit": 50}
inputs = {"input": "Who is the current captain of IPL team Mumbai Indians?"}


def main():
    for event in app.stream(inputs, config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    main()
