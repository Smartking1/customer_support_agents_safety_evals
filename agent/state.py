from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class ActionLogEntry(TypedDict):
    tool: str
    arguments: dict
    result: str


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: str
    orders_in_scope: list[dict]
    ticket_status: str
    action_log: list[ActionLogEntry]
    iteration_count: int
