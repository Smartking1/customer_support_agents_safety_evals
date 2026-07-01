"""CLI: run one support conversation against the agent."""
import sys
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import build_graph
from agent.prompts import get_system_prompt
from agent.tools import get_customer_order_ids


def run(customer_id: str, message: str) -> str:
    graph = build_graph()
    order_ids = get_customer_order_ids(customer_id)
    initial_state = {
        "messages": [
            SystemMessage(content=get_system_prompt(customer_id=customer_id, order_ids=order_ids)),
            HumanMessage(content=message),
        ],
        "customer_id": customer_id,
        "orders_in_scope": [],
        "ticket_status": "open",
        "action_log": [],
        "iteration_count": 0,
    }
    result = graph.invoke(initial_state)
    return result["messages"][-1].content


if __name__ == "__main__":
    customer_id = sys.argv[1] if len(sys.argv) > 1 else "CUST-A"
    message = sys.argv[2] if len(sys.argv) > 2 else "Can you look up my order ORD-1001?"
    print(f"\nCustomer ({customer_id}): {message}\n")
    reply = run(customer_id, message)
    print(f"Agent: {reply}\n")
