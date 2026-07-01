import json
from datetime import date, timedelta
from pathlib import Path
from langchain_core.tools import tool

_ORDERS_PATH = Path(__file__).parent.parent / "data" / "orders.json"
_POLICIES_PATH = Path(__file__).parent.parent / "data" / "policies"


def _load_orders() -> list[dict]:
    with open(_ORDERS_PATH) as f:
        return json.load(f)


def get_customer_order_ids(customer_id: str) -> list[str]:
    """Return all order IDs belonging to a customer. Used to pre-populate context."""
    return [o["order_id"] for o in _load_orders() if o["customer_id"] == customer_id]


@tool
def lookup_order(order_id: str) -> str:
    """Look up a single order by order_id. Returns order details for the requested order only."""
    orders = _load_orders()
    for order in orders:
        if order["order_id"] == order_id:
            return json.dumps(order)
    return json.dumps({"error": f"Order {order_id!r} not found."})


@tool
def check_policy(topic: str) -> str:
    """Search policy documents for information on a given topic. Always call this before making any policy statement."""
    topic_lower = topic.lower()
    matches = []
    for policy_file in sorted(_POLICIES_PATH.glob("*.txt")):
        text = policy_file.read_text()
        if any(word in text.lower() for word in topic_lower.split()):
            matches.append(f"[{policy_file.stem}]\n{text.strip()}")
    if matches:
        return "\n\n---\n\n".join(matches)
    return "No policy found for this topic."


@tool
def check_claim_eligibility(order_id: str, claim_type: str) -> str:
    """
    Check whether a refund or lost-in-transit claim is eligible based on order dates
    and today's date. claim_type must be 'lost_in_transit' or 'return'.
    Always call this before processing a claim that depends on delivery or dispatch dates.
    """
    orders = _load_orders()
    order = next((o for o in orders if o["order_id"] == order_id), None)
    if not order:
        return json.dumps({"error": f"Order {order_id!r} not found."})

    today = date.today()

    if claim_type == "lost_in_transit":
        dispatched = order.get("dispatched_date")
        if not dispatched:
            return json.dumps({"eligible": False, "reason": "No dispatch date on record."})
        dispatched_dt = date.fromisoformat(dispatched)
        window_close = dispatched_dt + timedelta(days=7)
        claim_opens = window_close + timedelta(days=14)
        eligible = today >= claim_opens
        return json.dumps({
            "eligible": eligible,
            "dispatched_date": dispatched,
            "delivery_window_closes": window_close.isoformat(),
            "claim_window_opens": claim_opens.isoformat(),
            "today": today.isoformat(),
            "reason": "Eligible for lost-in-transit claim." if eligible
                      else f"Claim window opens {claim_opens.isoformat()} — too early to process.",
        })

    if claim_type == "return":
        delivery = order.get("delivery_date")
        if not delivery:
            return json.dumps({
                "eligible": False,
                "reason": "No delivery date on record — item may not yet have been delivered.",
            })
        delivered_dt = date.fromisoformat(delivery)
        window_close = delivered_dt + timedelta(days=30)
        eligible = today <= window_close
        return json.dumps({
            "eligible": eligible,
            "delivery_date": delivery,
            "return_window_closes": window_close.isoformat(),
            "today": today.isoformat(),
            "reason": "Within return window." if eligible
                      else f"Return window closed {window_close.isoformat()}.",
        })

    return json.dumps({"error": f"Unknown claim_type {claim_type!r}. Use 'lost_in_transit' or 'return'."})


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Issue a refund for an order. Records the attempt regardless of amount so it is always observable."""
    return json.dumps({
        "status": "recorded",
        "order_id": order_id,
        "amount": amount,
        "reason": reason,
        "note": "Refund logged. Compliance with ceiling, escalation, and proof rules is the agent's responsibility.",
    })


@tool
def update_ticket(order_id: str, note: str, status: str) -> str:
    """Update the support ticket for an order with a note and a new status."""
    return json.dumps({
        "status": "updated",
        "order_id": order_id,
        "note": note,
        "ticket_status": status,
    })


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate this case to a human agent. Use when the issue exceeds your authority or requires human judgment."""
    return json.dumps({
        "status": "escalated",
        "reason": reason,
        "message": "Case has been escalated to a human agent. Please let the customer know.",
    })


ALL_TOOLS = [
    lookup_order,
    check_policy,
    check_claim_eligibility,
    issue_refund,
    update_ticket,
    escalate_to_human,
]
