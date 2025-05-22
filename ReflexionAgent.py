import os
# Ensure API keys are set, if not already handled by the environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "") # Add default empty string if not found
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "") # Add default empty string if not found

from typing import TypedDict, List, Tuple, operator, Annotated 

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Added MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_functions import create_structured_output_runnable 
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent, AgentExecutor

class ReflexionState(TypedDict):
    """
    Represents the state of the Reflexion agent.
    """
    input: str  # The user's query
    plan: List[str]  # The current plan
    executed_steps: Annotated[List[Tuple[str, str]], operator.add]  # Updated for accumulation
    reflections: Annotated[List[str], operator.add]  # Updated for accumulation
    response: str  # The final response to the user
    num_revisions: int  # Number of revisions or retries

class Plan(BaseModel):
    """Plan to achieve the objective."""
    steps: List[str] = Field(
        default=[],
        description="List of steps to achieve the objective, in order."
    )

PLANNER_PROMPT = """For the given objective, devise a concise, step-by-step plan. Each step should be a clear action. The final step's result should directly answer the objective. Ensure each step logically follows the previous one and contains all necessary information.

Objective: {objective}"""

prompt_template = ChatPromptTemplate.from_template(PLANNER_PROMPT)

# Assuming OPENAI_API_KEY is set in the environment
llm = ChatOpenAI(model="gpt-3.5-turbo")

# --- Executor Agent with Tavily Search Tool ---
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

EXECUTOR_AGENT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that executes tasks. You have access to tools. Use them if necessary to get the information to complete the task. If the task is a question, provide a concise answer. If the task is an action, describe the outcome."),
    MessagesPlaceholder(variable_name="chat_history", optional=True), 
    ("human", """Task: {input}
Previous actions and observations: {previous_steps}
Reflection on last step: {reflection}""")
])

executor_agent_runnable = create_openai_functions_agent(llm, tools, EXECUTOR_AGENT_PROMPT_TEMPLATE)
executor_agent = AgentExecutor(agent=executor_agent_runnable, tools=tools, verbose=False) # Set verbose=True for debugging if needed

# --- Planner ---
planner = create_structured_output_runnable(Plan, llm, prompt_template)

def plan_step(state: ReflexionState) -> dict:
    """
    Generates a plan based on the user's input.
    """
    objective = state["input"]
    plan_result = planner.invoke({"objective": objective})
    return {"plan": plan_result.steps}

def execute_step(state: ReflexionState) -> dict:
    """
    Executes the current task in the plan and returns the observation.
    """
    if not state["plan"]:
        # Should not happen if graph logic is correct (e.g., a condition to check if plan is empty)
        return {"response": "No plan to execute."}

    task = state["plan"][0]
    
    previous_steps_str = "\n".join([f"- Action: {action}, Observation: {observation}" for action, observation in state.get("executed_steps", [])])
    last_reflection = state.get("reflections", [])[-1] if state.get("reflections") else "No reflection on previous step."

    # Input for the executor agent
    agent_input = {
        "input": task,
        "previous_steps": previous_steps_str if previous_steps_str else "No steps executed yet.",
        "reflection": last_reflection
    }
    # Example for chat_history if EXECUTOR_AGENT_PROMPT_TEMPLATE uses it:
    # agent_input["chat_history"] = [HumanMessage(content=f"Objective: {state['input']}")]

    # Invoke the executor agent
    response = executor_agent.invoke(agent_input)
    observation = response.get("output", "No output from agent.")

    return {
        "executed_steps": [(task, observation)], # Return as a list of tuples to be appended by the graph
        "plan": state["plan"][1:] 
    }

REFLECTOR_PROMPT = """You are a self-reflection module. Your purpose is to analyze the execution of a plan and provide constructive feedback.
Objective: {objective}
Current Plan: {plan}
Last Executed Action: {action}
Observation from Last Action: {observation}

Based on the objective, the current plan, and the last action/observation, reflect on the following:
1.  Is the last action's observation directly helping to achieve the overall objective?
2.  Is the observation problematic, unclear, or insufficient?
3.  Does the current plan still seem like the best way to achieve the objective?
4.  Should the plan be revised, or should we try the last step again with a different approach, or is everything on track?

Provide a concise reflection. If you think the objective is met and we have a final answer, start your reflection with "CONCLUSION:". If you think the plan needs revision, start with "REVISE_PLAN:". If you think the last step needs to be retried (e.g. with a different prompt or tool), start with "RETRY_STEP:". Otherwise, if things are on track, start with "PROCEED:"."""

def reflect_step(state: ReflexionState) -> dict:
    objective = state["input"]
    current_plan_str = "\n".join(state["plan"]) if state["plan"] else "No further steps in plan."
    
    # Ensure executed_steps is not empty before accessing its last element
    if not state.get("executed_steps"): # Check if the key exists and is not empty
        # This case should ideally not be reached if reflect_step is called after an execution
        return {"reflections": ["No action taken yet to reflect upon."]}

    last_action, last_observation = state["executed_steps"][-1]

    reflector_prompt_text = REFLECTOR_PROMPT.format(
        objective=objective,
        plan=current_plan_str,
        action=last_action,
        observation=last_observation
    )
    
    # Assuming 'llm' is accessible (defined globally in the module)
    response = llm.invoke(reflector_prompt_text)
    reflection_text = response.content

    return {"reflections": [reflection_text]} # Return as a list to be appended

# --- Graph Definition ---

MAX_REVISIONS = 3

def prepare_for_replan(state: ReflexionState) -> dict:
    """
    Increments the number of revisions and prepares the state for replanning.
    """
    print("---PREPARING FOR REPLAN---")
    return {"num_revisions": state["num_revisions"] + 1}

def prepare_for_retry(state: ReflexionState) -> dict:
    """
    Increments the number of revisions and prepares the state for retrying the last step.
    """
    print("---PREPARING FOR RETRY---")
    # The plan is not modified here; executor will take the first step from the current plan
    return {"num_revisions": state["num_revisions"] + 1}

def extract_response(state: ReflexionState) -> dict:
    """
    Extracts the final response from the state.
    """
    print("---EXTRACTING RESPONSE---")
    if state.get("reflections") and state["reflections"][-1].startswith("CONCLUSION:"):
        final_answer = state["reflections"][-1][len("CONCLUSION:"):].strip()
    elif state.get("executed_steps"):
        # Use the last observation as a fallback
        final_answer = state["executed_steps"][-1][1] 
    else:
        final_answer = "Could not determine a final answer after processing."
    return {"response": final_answer}

def should_continue(state: ReflexionState) -> str:
    """
    Determines the next step based on the current state and reflections.
    """
    print("---DECIDING NEXT STEP---")
    if not state.get("reflections"): 
        print("No reflections found. Ending.")
        return "extract_response" # Should ideally not happen if called after reflector
    
    last_reflection = state["reflections"][-1]
    print(f"Last Reflection: {last_reflection}")
    print(f"Number of Revisions: {state['num_revisions']}")
    print(f"Current Plan: {state['plan']}")

    if state["num_revisions"] >= MAX_REVISIONS:
        print(f"Max revisions ({MAX_REVISIONS}) reached. Extracting response.")
        return "extract_response"

    if last_reflection.startswith("CONCLUSION:"):
        print("Reflection indicates CONCLUSION.")
        return "extract_response"
    elif last_reflection.startswith("REVISE_PLAN:"):
        print("Reflection suggests REVISE_PLAN.")
        return "prepare_for_replan"
    elif last_reflection.startswith("RETRY_STEP:"):
        if not state["plan"]:
            print("Warning: RETRY_STEP requested with an empty plan. Revising plan instead.")
            return "prepare_for_replan" 
        print("Reflection suggests RETRY_STEP.")
        return "prepare_for_retry"
    elif state["plan"]: # "PROCEED:" and plan has steps
        print("Reflection suggests PROCEED and plan has steps.")
        return "executor"
    else: # "PROCEED:" but no plan, or unhandled case
        print("Proceeding to extract response as plan is empty or reflection is unspecific.")
        return "extract_response"

# Initialize the StateGraph
workflow = StateGraph(ReflexionState)

# Add nodes
workflow.add_node("planner", plan_step)
workflow.add_node("executor", execute_step)
workflow.add_node("reflector", reflect_step)
workflow.add_node("prepare_for_replan", prepare_for_replan)
workflow.add_node("prepare_for_retry", prepare_for_retry)
workflow.add_node("extract_response", extract_response)

# Set the entry point
workflow.set_entry_point("planner")

# Add edges
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reflector")

# Add conditional edges from reflector
workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "prepare_for_replan": "prepare_for_replan",
        "prepare_for_retry": "prepare_for_retry",
        "executor": "executor",
        "extract_response": "extract_response", # Ensure this path exists in should_continue returns
    }
)

# Add edges from preparation nodes
workflow.add_edge("prepare_for_replan", "planner")
workflow.add_edge("prepare_for_retry", "executor")
workflow.add_edge("extract_response", END)

# Compile the graph
app = workflow.compile()

# Example of how to run (optional, for testing purposes, will be in a separate file)
# if __name__ == "__main__":
#     initial_state = ReflexionState(
#         input="What is the capital of France?",
#         plan=[],
#         executed_steps=[],
#         reflections=[],
#         response="",
#         num_revisions=0
#     )
#     final_state = app.invoke(initial_state)
#     print("\n---FINAL RESPONSE---")
#     print(final_state.get("response"))

if __name__ == "__main__":
    # Example usage
    initial_input = {"input": "What is the main concept behind the ReWOO agent framework and who created it?", "num_revisions": 0} 
    # Ensure 'num_revisions' is part of the initial input for the state.
    # Other fields like plan, executed_steps, reflections, response will be initialized as empty or default by the graph if not provided.
    # However, to be explicit with ReflexionState, we can define them:
    initial_input.update({
        "plan": [],
        "executed_steps": [],
        "reflections": [],
        "response": ""
    })


    print(f"Starting Reflexion Agent with input: {initial_input['input']}\n")

    # Stream events from the graph
    for event_counter, event_data in enumerate(app.stream(initial_input, {"recursion_limit": 100})):
        print(f"--- Event {event_counter + 1} ---")
        for key, value in event_data.items():
            # __end__ is a special key indicating the end of the graph stream
            if key == "__end__":
                print("\n--- Agent Finished ---")
                # The final response should be in the 'response' field of the last relevant state
                # This part might need adjustment based on how 'extract_response' and END node work with final state.
                # Typically, the value associated with '__end__' might be the final state itself.
                final_state_from_stream = value 
                print(f"Final Response (from stream's __end__): {final_state_from_stream.get('response', 'N/A')}")
            else:
                print(f"Node '{key}':")
                # Ensure complex objects like 'executed_steps' or 'reflections' are printed concisely if needed
                if isinstance(value, dict) and value.get('executed_steps'):
                    print(f"  Output (executed_steps update): {value['executed_steps']}")
                elif isinstance(value, dict) and value.get('reflections'):
                     print(f"  Output (reflections update): {value['reflections']}")
                elif isinstance(value, dict) and value.get('plan'):
                     print(f"  Output (plan update): {value['plan']}")
                elif isinstance(value, dict) and value.get('response'):
                     print(f"  Output (response update): {value['response']}")
                else:
                    print(f"  Output: {value}") # Fallback for other types of values
        print("---------------------\n")

    # As an alternative or for clearer final output, you can also invoke the graph 
    # and get the final state, then print specific parts of it.
    # print("\n--- Invoking Agent for Final State (Alternative) ---")
    # final_state_invoke = app.invoke(initial_input, {"recursion_limit": 100})
    # print("\n--- Agent Finished (Invoke) ---")
    # print(f"Final Input: {final_state_invoke.get('input')}")
    # print(f"Final Plan: {final_state_invoke.get('plan')}")
    # print(f"Executed Steps: {final_state_invoke.get('executed_steps')}")
    # print(f"Reflections: {final_state_invoke.get('reflections')}")
    # print(f"Number of Revisions: {final_state_invoke.get('num_revisions')}")
    # print(f"Final Response: {final_state_invoke.get('response', 'N/A')}")
