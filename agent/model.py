import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_model():
    """Return the configured LangChain chat model. Provider/name come from env or config.yaml."""
    cfg = _load_config()
    provider = os.getenv("MODEL_PROVIDER", cfg["model"]["provider"]).lower()
    name = os.getenv("MODEL_NAME", cfg["model"]["name"])

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=name, temperature=0)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=name)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, temperature=0)

    raise ValueError(f"Unknown MODEL_PROVIDER: {provider!r}. Choose groq, anthropic, or openai.")


def get_refund_ceiling() -> float:
    cfg = _load_config()
    return float(cfg["agent"]["refund_ceiling"])


def get_max_iterations() -> int:
    cfg = _load_config()
    return int(cfg["agent"]["max_iterations"])
