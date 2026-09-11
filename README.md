# LakVeggie AI

Natural-language questions over the live LakVeggie product catalogue (MySQL), answered two different ways so the two approaches can be compared side by side.

| | `app.py` — text-to-SQL chain | `agent.py` — tool-calling agent |
|---|---|---|
| Control flow | question → write SQL → run → phrase answer | question → model picks a tool → sees result → decides again → … → answer |
| LLM calls | always 2 | 1 to N |
| Who writes the SQL | the model (validated afterwards) | we do; the model only supplies tool arguments |
| Port | 8000 | 8001 |

`console.html` is a static page that queries both services at once and shows the answers, the generated SQL, and the agent's tool trace in parallel lanes.

## How it works

### Text-to-SQL chain (`app.py`)

Two prompts in sequence: one turns the question into a MySQL `SELECT`, the other turns the result rows into a sentence. Between them sits `sanitize_sql()`, which rejects anything that is not a single `SELECT`/`WITH`, blocks write and file keywords, blocks references to tables outside the allow list, and appends `LIMIT 50` when the model forgets one.

The prompt encodes three facts about the `products` table that otherwise produce wrong answers:

- `price` is a `VARCHAR`, so `'150'` sorts before `'60'`. The generated column `price_lkr` holds the numeric copy — all sorting, comparison and arithmetic must use it.
- Sinhala values in `category` were corrupted to literal `?` characters, so the model must never put Sinhala text into SQL. It translates the vegetable name to English first.
- `type` and `category` are inconsistently filled and the table contains non-vegetables (rice), so filtering on them silently drops rows. Filter on `name` instead.

### Tool-calling agent (`agent.py`)

A LangGraph ReAct agent over five hand-written tools:

| Tool | Purpose |
|---|---|
| `list_all_product_names` | exact catalogue spellings — called first when a name is uncertain or given in Sinhala |
| `get_product_details` | price, stock and unit for a name substring |
| `get_low_stock_products` | what needs restocking, most urgent first |
| `get_catalogue_summary` | totals, out-of-stock count, cheapest/most expensive |
| `calculate_margin` | profit arithmetic, done in Python rather than in the model's head |
| `search_market_price` | *optional* — wholesale prices via DuckDuckGo; registers only if `ddgs` is installed |

Every tool routes through one parameterised-query helper, so the query text is always written by us. The response carries `steps`, `elapsed_s` and a full `trace` of tool calls and results, plus a `warning` when the agent repeats an identical call three times or hits the step ceiling.

## Security model

The cPanel MySQL user has `SELECT` on the **whole** database — users, orders and purchase included — because cPanel cannot grant per-table privileges. Two layers stand in for that missing grant:

1. `ALLOWED_TABLES = ["products"]` is the only schema the LLM is ever shown.
2. `BANNED_TABLES` rejects generated SQL that names anything else, even a syntactically valid `SELECT`.

The threat being defended against is prompt injection, not a buggy model. Do not widen either list without thinking about what leaves the building.

The agent needs no such guard: the model never supplies SQL at all.

## Setup

Requires Python 3.11+, a running [Ollama](https://ollama.com) with `qwen3:8b` pulled, and SSH access to the database host.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
```

Create a `.env` in the project root — it is gitignored and must never be committed:

```
DB_URL=mysql+pymysql://<user>:<password>@127.0.0.1:3307/<database>
```

`OPENAI_API_KEY` / `GEMINI_API_KEY` are only needed if you switch the model — both files have the `ChatOpenAI` and `ChatGoogleGenerativeAI` alternatives commented out next to the active `ChatOllama`.

## Running

The database is not exposed publicly; everything goes through an SSH tunnel that must stay open for the whole session.

```bash
# tab 1 — leave open
ssh -N -L 3307:127.0.0.1:3306 <user>@<host>

# tab 2 — text-to-SQL chain
uvicorn app:app --reload --port 8000

# tab 3 — tool-calling agent
uvicorn agent:app --reload --port 8001
```

Then open `console.html` in a browser. The two dots at the top go green when both services answer `/health`.

## API

Both services expose the same shape.

`GET /health` → `{"status": "ok", ...}`, or `503` when the tunnel is down.

`POST /ask`

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What is the price of carrot?"}'
```

Chain response:

```json
{ "answer": "Carrot is Rs. 60.00.", "sql": "SELECT name, price_lkr FROM products WHERE name LIKE '%carrot%' LIMIT 50", "rows": "[('Carrot', Decimal('60.00'))]" }
```

Agent response adds the trace:

```json
{ "answer": "...", "steps": 2, "elapsed_s": 4.31, "trace": [{"tool": "get_product_details", "args": {"name": "carrot"}, "result": "..."}], "warning": null }
```

Questions are 3–500 characters. The agent accepts an optional `max_steps` (1–25, default 12).

## Behaviour worth knowing

- Both services answer in the language of the question — Sinhala in, Sinhala out — but product names always stay in English, because that is how they are stored.
- There is no price history in the schema. Asked about trends or past prices, the chain emits `SELECT 'NO_HISTORY'` and the answer explains that only current prices are stored.
- Database prices are **retail**. Wholesale figures only ever come from `search_market_price`, and the agent is told to say when a number came from the web.
- CORS is wide open (`allow_origins=["*"]`) for local development. Lock it down before this runs anywhere else.

## Layout

```
app.py          text-to-SQL chain service      (port 8000)
agent.py        tool-calling agent service     (port 8001)
console.html    static side-by-side console
requirements.txt
```
