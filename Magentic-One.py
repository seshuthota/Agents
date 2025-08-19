import os
import config
from config import MODEL_NAME, BIG_MODEL_NAME
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

config_list = [{"model": MODEL_NAME, "api_key": os.getenv("OPENAI_API_KEY") or ""}]
big_config_list = [{"model": BIG_MODEL_NAME, "api_key": os.getenv("OPENAI_API_KEY") or ""}]

planner = AssistantAgent(
    "Planner",
    system_message="""You are a planner. Break the user request into research steps and request help from Researcher and Writer.""",
    llm_config={"config_list": big_config_list},
)

researcher = AssistantAgent(
    "Researcher",
    system_message="""You gather information using the web_search tool and summarize findings for Writer.""",
    llm_config={"config_list": config_list},
)

writer = AssistantAgent(
    "Writer",
    system_message="""You compose the final answer for the user using the gathered research.""",
    llm_config={"config_list": config_list},
)

critic = AssistantAgent(
    "Critic",
    system_message="""Review the Writer's answer for accuracy and completeness before termination.""",
    llm_config={"config_list": big_config_list},
)

user = UserProxyAgent(
    "User",
    human_input_mode="NEVER",
    code_execution_config=False,
)


def web_search(query: str) -> str:
    """Use DuckDuckGo to search the web and return the results."""
    return search.run(query)


researcher.register_function({"web_search": web_search})

groupchat = GroupChat(
    agents=[user, planner, researcher, writer, critic],
    messages=[],
    max_round=12,
)

manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})


if __name__ == "__main__":
    user.initiate_chat(
        manager,
        message="Provide a short profile of the 2024 Australian Open singles champions and their hometowns.",
    )
