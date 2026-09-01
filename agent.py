"""
LakVeggie procurement agent.

This is the AGENT counterpart to the chain in app.py. Same database, same
guardrail philosophy, completely different control flow:

    chain (app.py)   question -> write SQL -> run -> phrase answer.   Always 2 LLM calls.
    agent (this)     question -> [model picks a tool -> sees result -> decides
                                  again -> ...] -> answer.            1 to N LLM calls.

The point of running both is to MEASURE the difference, not to assume the
agent is better. Keep app.py working.

Run:
    # tab 1, leave open:
    ssh -N -L 3307:127.0.0.1:3306 ceyloz9@ceylonapz.com
    # tab 2:
    uvicorn agent:app --reload --port 8001
"""

import os
import re
import json
import time
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lakveggie-agent")

# --------------------------------------------------------------------- config

DB_URL = os.environ["DB_URL"]

MAX_STEPS = 12          # hard ceiling on agent iterations
MAX_REPEATS = 3         # same tool + same args this many times = stuck
TOOL_TIMEOUT_S = 20

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_recycle=280,
    pool_size=3,
    connect_args={"connect_timeout": 10},
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)


def _rows(sql: str, params: dict | None = None) -> list[dict]:
    """Run a parameterised read-only query and return plain dicts.

    Every tool goes through this. The model never supplies SQL, only tool
    arguments, so the query text is always written by us. That is a much
    stronger guarantee than validating model-written SQL after the fact.
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(r._mapping) for r in result]


# ----------------------------------------------------------------- the tools
#
# Tool design IS the engineering here. The name, the docstring and the
# parameter names are all sent to the model as its only description of what
# the tool does. When an agent picks the wrong tool, the docstring is usually
# at fault, not the model.

@tool
def list_all_product_names() -> list[str]:
    """List every product name in the LakVeggie catalogue, spelled exactly as
    stored in the database.

    Call this FIRST whenever you are unsure how a product is spelled, or when
    the user names something in Sinhala. Product names may contain spaces
    (for example 'Gotu Kola', 'Red Onion'). Never guess a spelling: look it up
    here, then use the exact string in other tools.
    """
    return [r["name"] for r in _rows("SELECT name FROM products ORDER BY name")]


@tool
def get_product_details(name: str) -> list[dict]:
    """Look up one or more products by name and return price, stock and unit.

    `name` is matched case-insensitively as a substring, so 'kola' matches
    'Gotu Kola'. Prefer a single distinctive word over a full multi-word name.
    Returns an empty list if nothing matches, which means the product is not
    in the catalogue.
    """
    return _rows(
        "SELECT id, name, price_lkr, quantity, unit_type "
        "FROM products WHERE name LIKE :pat LIMIT 20",
        {"pat": f"%{name}%"},
    )


@tool
def get_low_stock_products(threshold: int = 10) -> list[dict]:
    """List products whose stock is at or below `threshold` units.

    Use this to work out what needs restocking. A quantity of 0 means the
    product is completely out of stock. Results are ordered by quantity, most
    urgent first.
    """
    return _rows(
        "SELECT name, price_lkr, quantity FROM products "
        "WHERE quantity <= :t ORDER BY quantity ASC LIMIT 50",
        {"t": threshold},
    )


@tool
def get_catalogue_summary() -> dict:
    """Return overall catalogue statistics: total products, how many are out of
    stock, and the cheapest and most expensive current prices.

    Use this for broad questions about the catalogue as a whole, instead of
    listing every product.
    """
    return _rows(
        "SELECT COUNT(*) AS total, "
        "SUM(quantity = 0) AS out_of_stock, "
        "MIN(price_lkr) AS cheapest, "
        "MAX(price_lkr) AS most_expensive "
        "FROM products"
    )[0]


@tool
def calculate_margin(buy_price: float, sell_price: float, quantity: int) -> dict:
    """Calculate profit and margin for buying `quantity` units at `buy_price`
    and selling them at `sell_price`. All prices in LKR per unit.

    ALWAYS use this tool for profit arithmetic instead of computing it
    yourself. Returns a negative margin when the sell price is below the buy
    price, which is a signal worth reporting to the user.
    """
    cost = buy_price * quantity
    revenue = sell_price * quantity
    profit = revenue - cost
    margin_pct = (profit / revenue * 100) if revenue else 0.0
    return {
        "cost_lkr": round(cost, 2),
        "revenue_lkr": round(revenue, 2),
        "profit_lkr": round(profit, 2),
        "margin_pct": round(margin_pct, 2),
        "loss_making": profit < 0,
    }


TOOLS = [
    list_all_product_names,
    get_product_details,
    get_low_stock_products,
    get_catalogue_summary,
    calculate_margin,
]

# Optional: wholesale price lookup via web search. Needs `pip install ddgs`.
# Kept optional so the agent still runs without it.
try:
    from langchain_community.tools import DuckDuckGoSearchRun

    _search = DuckDuckGoSearchRun()

    @tool
    def search_market_price(vegetable: str) -> str:
        """Search the web for the current wholesale price of a vegetable at Sri
        Lankan economic centres (Dambulla, Meegoda, Keppetipola).

        Use this only when the user asks about wholesale, market or buying
        prices. The LakVeggie database holds RETAIL prices, not wholesale ones.
        Web results may be stale or wrong; say so when you use them.
        """
        return _search.invoke(
            f"{vegetable} wholesale price today Dambulla economic centre Sri Lanka"
        )[:1500]

    TOOLS.append(search_market_price)
    log.info("web search tool enabled")
except Exception as exc:                          # noqa: BLE001
    log.warning("web search tool disabled: %s", exc)


# ---------------------------------------------------------------- the prompt

SYSTEM_PROMPT = """You are the LakVeggie procurement assistant. You help the \
owner decide what to restock and at what price.

How to work:
- Use the tools. Never state a price, a stock level or a product name that did \
not come from a tool result.
- If you are unsure how a product is spelled, call list_all_product_names \
before anything else. Product names contain spaces.
- The user may write in Sinhala. Translate vegetable names to English yourself, \
then confirm the spelling against the catalogue.
- Database prices are RETAIL prices. Wholesale prices come only from web search.
- Use calculate_margin for any profit arithmetic. Do not do it in your head.
- If a tool returns nothing, say so plainly. Do not substitute a guess.
- If a result looks wrong (a retail price below a plausible wholesale price, \
for instance), say so instead of quietly reporting it.
- Stop as soon as you can answer. Do not call tools you do not need.

Reply in the language the user asked in. Keep product names in English. \
Format prices as 'Rs. 60.00'. Be brief."""

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


# ------------------------------------------------------------ loop detection

def _tool_calls(messages: list[Any]) -> list[dict]:
    """Pull a flat list of {tool, args, result} out of the message history."""
    calls: list[dict] = []
    pending: dict[str, dict] = {}

    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            entry = {"tool": tc["name"], "args": tc["args"], "result": None}
            pending[tc["id"]] = entry
            calls.append(entry)
        if getattr(m, "type", None) == "tool":
            entry = pending.get(getattr(m, "tool_call_id", ""))
            if entry is not None:
                entry["result"] = str(m.content)[:400]
    return calls

def _text(content: Any) -> str:
    """Flatten a message's content to plain text.

    Reasoning models return a list of content blocks rather than a string, so
    a bare .content is not safe to hand to Pydantic.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p).strip()
    return str(content)

def _detect_loop(calls: list[dict]) -> str | None:
    """Flag the same tool being called with identical arguments repeatedly.

    A stuck agent burns tokens silently. Better to surface it than to let the
    recursion limit end the run with no explanation.
    """
    seen: dict[str, int] = {}
    for c in calls:
        key = f"{c['tool']}::{json.dumps(c['args'], sort_keys=True, default=str)}"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= MAX_REPEATS:
            return f"Repeated {c['tool']} with identical arguments {seen[key]} times."
    return None


# ----------------------------------------------------------------- FastAPI

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    max_steps: int = Field(MAX_STEPS, ge=1, le=25)


class AskResponse(BaseModel):
    answer: str
    steps: int
    elapsed_s: float
    trace: list[dict]
    warning: str | None = None


app = FastAPI(title="LakVeggie Agent", version="0.1.0")


@app.get("/health")
def health():
    try:
        _rows("SELECT 1 AS ok")
        return {"status": "ok", "tools": [t.name for t in TOOLS]}
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(503, f"Database unreachable. Is the tunnel up? {exc}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    started = time.perf_counter()

    try:
        state = agent.invoke(
            {"messages": [("user", req.question)]},
            config={"recursion_limit": req.max_steps * 2},
        )
    except Exception as exc:                      # noqa: BLE001
        log.exception("agent run failed")
        raise HTTPException(500, f"Agent failed: {exc}") from exc

    messages = state["messages"]
    calls = _tool_calls(messages)
    answer = _text(messages[-1].content) if messages else ""

    warning = _detect_loop(calls)
    if len(calls) >= req.max_steps:
        warning = (warning or "") + f" Hit the step ceiling ({req.max_steps})."

    log.info("%d tool calls in %.1fs", len(calls), time.perf_counter() - started)

    return AskResponse(
        answer=answer,
        steps=len(calls),
        elapsed_s=round(time.perf_counter() - started, 2),
        trace=calls,
        warning=warning.strip() if warning else None,
    )