import json
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.model import get_model, get_max_iterations
from agent.tools import ALL_TOOLS


def _build_tool_map():
    return {t.name: t for t in ALL_TOOLS}


def model_node(state: AgentState) -> dict:
    model = get_model().bind_tools(ALL_TOOLS)
    response = model.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tool_node(state: AgentState) -> dict:
    tool_map = _build_tool_map()
    last_message = state["messages"][-1]
    action_log = list(state.get("action_log", []))
    new_messages = []
    ticket_status = state.get("ticket_status", "open")

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_fn = tool_map.get(tool_name)

        if tool_fn:
            result = tool_fn.invoke(tool_args)
        else:
            result = json.dumps({"error": f"Unknown tool: {tool_name}"})

        # Data isolation guard: block lookup_order results that belong to another customer
        if tool_name == "lookup_order":
            try:
                result_data = json.loads(result)
                order_customer = result_data.get("customer_id")
                if order_customer and order_customer != state.get("customer_id"):
                    result = json.dumps({"error": "This order is not associated with your account."})
            except Exception:
                pass

        action_log.append({
            "tool": tool_name,
            "arguments": tool_args,
            "result": result,
        })
        new_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

        # Keep ticket_status in state in sync with update_ticket calls
        if tool_name == "update_ticket":
            try:
                ticket_status = tool_call["args"].get("status", ticket_status)
            except Exception:
                pass

    return {"messages": new_messages, "action_log": action_log, "ticket_status": ticket_status}


def _should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if state.get("iteration_count", 0) >= get_max_iterations():
        return END
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()
