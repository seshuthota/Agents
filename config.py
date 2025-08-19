import json
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
load_dotenv()

with open(os.path.join(BASE_DIR, "config.json")) as f:
    _config = json.load(f)

LANGCHAIN_TRACING_V2 = str(
    os.getenv("LANGCHAIN_TRACING_V2", _config.get("LANGCHAIN_TRACING_V2", True))
).lower()
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", _config.get("LANGCHAIN_PROJECT", "Agents"))
MODEL_NAME = os.getenv("MODEL_NAME", _config.get("MODEL_NAME", "gpt-5-2025-08-07"))
BIG_MODEL_NAME = os.getenv("BIG_MODEL_NAME", _config.get("BIG_MODEL_NAME", MODEL_NAME))

os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)
if os.getenv("LS_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LS_API_KEY")
