"""
Support-agent safety eval runner.

Usage:
    # Run all scenarios against the default model (groq/llama-3.3-70b-versatile):
    python -m evals.run_eval

    # Run against a different model:
    python -m evals.run_eval --model groq/llama-3.1-8b-instant
    python -m evals.run_eval --model anthropic/claude-haiku-4-5-20251001

    # Run a specific scenario only:
    python -m evals.run_eval --scenario social_pressure
    python -m evals.run_eval --scenario policy_fabrication
    python -m evals.run_eval --scenario prompt_injection
    python -m evals.run_eval --scenario data_leakage
    python -m evals.run_eval --scenario out_of_policy_action
"""
import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv

# Load .env FIRST so MODEL_PROVIDER / MODEL_NAME from the file are available
# before any other code reads or overwrites them.
load_dotenv()


# ── Model helper ────────────────────────────────────────────────────────────────

def _set_model_env(model_str: str):
    """Parse 'provider/model-name' and write to env so the agent picks it up."""
    if "/" in model_str:
        provider, name = model_str.split("/", 1)
    else:
        provider, name = "groq", model_str
    os.environ["MODEL_PROVIDER"] = provider
    os.environ["MODEL_NAME"] = name
    return provider, name


def _read_model_env() -> tuple[str, str]:
    """Read the current provider/name from env (already populated by load_dotenv)."""
    provider = os.environ.get("MODEL_PROVIDER", "anthropic")
    name = os.environ.get("MODEL_NAME", "claude-haiku-4-5-20251001")
    return provider, name


# ── Solver ───────────────────────────────────────────────────────────────────────

from inspect_ai.solver import solver, TaskState, Generate, Solver

# Serialize agent calls to respect Groq's free-tier TPM limit
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(1)
    return _semaphore


@solver
def langgraph_agent_solver() -> Solver:
    """Wraps the LangGraph agent as an Inspect solver."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        from inspect_ai.model import ChatMessageAssistant
        from agent.graph import build_graph
        from agent.prompts import get_system_prompt
        from agent.tools import get_customer_order_ids

        customer_id = state.metadata.get("customer_id", "CUST-UNKNOWN")
        order_ids = get_customer_order_ids(customer_id)

        # Build conversation history from Inspect's state messages.
        # Supports both single-turn (one user message) and multi-turn (list input).
        def _to_str(content) -> str:
            if isinstance(content, str):
                return content
            return " ".join(p.text if hasattr(p, "text") else str(p) for p in content)

        lc_messages = [SystemMessage(content=get_system_prompt(customer_id=customer_id, order_ids=order_ids))]
        for msg in state.messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=_to_str(msg.content)))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=_to_str(msg.content)))

        if not any(isinstance(m, HumanMessage) for m in lc_messages):
            return state

        graph = build_graph()
        agent_input = {
            "messages": lc_messages,
            "customer_id": customer_id,
            "orders_in_scope": [],
            "ticket_status": "open",
            "action_log": [],
            "iteration_count": 0,
        }

        # Serialize calls + retry on rate-limit to handle Groq free-tier TPM cap
        async with _get_semaphore():
            for attempt in range(4):
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, graph.invoke, agent_input)
                    break
                except Exception as exc:
                    if "rate_limit" in str(exc).lower() and attempt < 3:
                        wait = 15 * (2 ** attempt)
                        print(f"  [rate limit] waiting {wait}s before retry {attempt + 1}/3")
                        await asyncio.sleep(wait)
                    else:
                        raise

        # Surface the action log so scorers can inspect it
        state.metadata["action_log"] = result["action_log"]

        # Append the agent's final reply to the conversation
        final_reply = result["messages"][-1].content
        state.messages.append(ChatMessageAssistant(content=final_reply))

        # Set state.output so Inspect's viewer displays the agent's response
        from inspect_ai.model import ModelOutput
        model_id = f"{os.environ.get('MODEL_PROVIDER', 'anthropic')}/{os.environ.get('MODEL_NAME', 'unknown')}"
        state.output = ModelOutput.from_content(model=model_id, content=final_reply)

        return state

    return solve


# ── Tasks ────────────────────────────────────────────────────────────────────────

from inspect_ai import Task, task
from evals.scenarios.social_pressure import social_pressure_samples
from evals.scenarios.policy_fabrication import policy_fabrication_samples
from evals.scenarios.prompt_injection import prompt_injection_samples
from evals.scenarios.data_leakage import data_leakage_samples
from evals.scenarios.out_of_policy_action import out_of_policy_action_samples
from evals.scenarios.multi_turn import multi_turn_samples
from evals.scorers import (
    social_pressure_score,
    policy_fabrication_score,
    prompt_injection_score,
    data_leakage_score,
    out_of_policy_score,
    multi_turn_score,
)


@task
def social_pressure():
    return Task(
        dataset=social_pressure_samples(),
        solver=langgraph_agent_solver(),
        scorer=social_pressure_score(),
    )


@task
def policy_fabrication():
    return Task(
        dataset=policy_fabrication_samples(),
        solver=langgraph_agent_solver(),
        scorer=policy_fabrication_score(),
    )


@task
def prompt_injection():
    return Task(
        dataset=prompt_injection_samples(),
        solver=langgraph_agent_solver(),
        scorer=prompt_injection_score(),
    )


@task
def data_leakage():
    return Task(
        dataset=data_leakage_samples(),
        solver=langgraph_agent_solver(),
        scorer=data_leakage_score(),
    )


@task
def out_of_policy_action():
    return Task(
        dataset=out_of_policy_action_samples(),
        solver=langgraph_agent_solver(),
        scorer=out_of_policy_score(),
    )


@task
def multi_turn():
    return Task(
        dataset=multi_turn_samples(),
        solver=langgraph_agent_solver(),
        scorer=multi_turn_score(),
    )


_ALL_TASKS = [
    social_pressure,
    policy_fabrication,
    prompt_injection,
    data_leakage,
    out_of_policy_action,
    multi_turn,
]

_SCENARIO_MAP = {
    "social_pressure": [social_pressure()],
    "policy_fabrication": [policy_fabrication()],
    "prompt_injection": [prompt_injection()],
    "data_leakage": [data_leakage()],
    "out_of_policy_action": [out_of_policy_action()],
    "multi_turn": [multi_turn()],
    "all": [t() for t in _ALL_TASKS],
}


# ── CLI runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run support-agent safety eval")
    parser.add_argument(
        "--model",
        default=None,
        help="Agent model in provider/name format (e.g. anthropic/claude-sonnet-4-6). "
             "Defaults to MODEL_PROVIDER/MODEL_NAME from .env or config.yaml.",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        choices=list(_SCENARIO_MAP.keys()),
        help="Scenario(s) to run (social_pressure, policy_fabrication, prompt_injection, data_leakage, out_of_policy_action, multi_turn, all)",
    )
    args = parser.parse_args()

    # --model flag overrides .env; omitting it uses .env / config.yaml values
    if args.model:
        provider, name = _set_model_env(args.model)
    else:
        provider, name = _read_model_env()
    print(f"\nRunning eval  model={provider}/{name}  scenario={args.scenario}\n")

    from inspect_ai import eval as inspect_eval

    results = inspect_eval(
        _SCENARIO_MAP[args.scenario],
        model=f"{provider}/{name}",
        log_dir="results/",
    )

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Model: {provider}/{name}")
    print(f"{'=' * 60}")
    for r in results:
        print(f"\nScenario : {r.eval.task}")
        print(f"Status   : {r.status}")
        if r.status == "success" and r.results and r.results.metrics:
            for metric, val in r.results.metrics.items():
                print(f"  {metric:12s}: {val.value:.3f}")
        elif r.error:
            print(f"  ERROR: {r.error.message}")
