"""
LakVeggie AI: natural language questions over the live LakVeggie MySQL database.

Connects through an SSH tunnel on 127.0.0.1:3307 using a SELECT-only MySQL user.

Run:
    # tab 1, leave open for the whole session:
    ssh -N -L 3307:127.0.0.1:3306 ceyloz9@ceylonapz.com
    # tab 2:
    uvicorn app:app --reload --port 8000
"""

import os
import re
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lakveggie-ai")

# --------------------------------------------------------------------- config

DB_URL = os.environ["DB_URL"]

# SECURITY: the cPanel user has SELECT on the WHOLE database, including users,
# orders and purchase. cPanel cannot grant per-table, so THIS LIST is what
# actually keeps customer data out of the LLM's reach. Do not widen it without
# thinking about what leaves the building.
ALLOWED_TABLES = ["products"]
MAX_ROWS = 50

engine = create_engine(
    DB_URL,
    pool_pre_ping=True,   # the SSH tunnel drops; detect dead connections
    pool_recycle=280,     # recycle before MySQL's wait_timeout kills them
    pool_size=3,
    max_overflow=2,
    connect_args={"connect_timeout": 10},
)

db = SQLDatabase(
    engine,
    include_tables=ALLOWED_TABLES,
    sample_rows_in_table_info=3,
)

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    reasoning=False,
    num_predict=1024,        # tokens 1024කින් නවත්තනවා
    num_ctx=8192,            # 32k ඕනේ නැහැ, RAM බේරෙනවා
)

# ----------------------------------------------------------------- SQL guard

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"replace|call|load_file|into\s+outfile)\b",
    re.IGNORECASE,
)

# Second layer: even a valid SELECT must not reach a table outside the allow
# list. Prompt injection is the threat here, not a buggy model.
BANNED_TABLES = re.compile(
    r"\b(users|orders|purchase|admin|tax|stores|outlets|sku|banner|offer|"
    r"city|application)\b",
    re.IGNORECASE,
)


def sanitize_sql(raw: str) -> str:
    sql = re.sub(r"^```(?:sql)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    sql = sql.rstrip(";").strip()

    if ";" in sql:
        raise HTTPException(400, "Only a single statement is allowed.")
    if not sql.lower().startswith(("select", "with")):
        raise HTTPException(400, "Only SELECT queries are allowed.")
    if FORBIDDEN.search(sql):
        raise HTTPException(400, "Query contains a forbidden keyword.")
    if BANNED_TABLES.search(sql):
        log.warning("Blocked attempt to reach a non-allowed table: %s", sql)
        raise HTTPException(400, "Query references a table that is not available.")
    if not re.search(r"\blimit\s+\d+", sql, re.IGNORECASE):
        sql = f"{sql} LIMIT {MAX_ROWS}"
    return sql


# ------------------------------------------------------------------- prompts
#
# Three facts about this table drive the prompt:
#   1. `price` is VARCHAR, so '150' sorts before '60'. A generated column
#      `price_lkr` holds the numeric copy; everything numeric must use it.
#   2. Sinhala values in `category` were corrupted to literal '?' characters,
#      so the model must never put Sinhala text into SQL.
#   3. `type` and `category` are inconsistently filled and the table contains
#      non-vegetables (rice), so filtering on them silently loses rows.

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write MySQL SELECT queries for the LakVeggie products table.\n\n"
     "Schema:\n{schema}\n\n"
     "Column meanings:\n"
     "- name: product name in English (e.g. Carrot, Beetroot, Cabbage)\n"
     "- price: display price as text. Never sort or compare on this column.\n"
     "- price_lkr: the same price as a DECIMAL. ALWAYS use price_lkr for "
     "sorting, comparing, filtering by amount and arithmetic.\n"
     "- quantity: units currently in stock. 0 means out of stock.\n"
     "- type and category: unreliable, inconsistently filled labels. The table "
     "also contains non-vegetable items such as rice. NEVER filter on type or "
     "category. Filter on the name column instead, or return all rows and let "
     "the user judge.\n\n"
     "Rules:\n"
     "- Output ONLY the SQL. No markdown fences, no explanation.\n"
     "- SELECT statements only, from the products table only.\n"
     "- Match product names case-insensitively with LIKE, e.g. "
     "WHERE name LIKE '%carrot%'.\n"
     "- 'In stock' or 'available' means quantity > 0.\n"
     "- The database holds English product names only. If the user asks in "
     "Sinhala, translate the vegetable name to English yourself before "
     "matching. Never put Sinhala text in the SQL.\n"
     "- Always select the name column so the answer can name the product.\n"
     f"- Never return more than {MAX_ROWS} rows.\n"
     "- There is no price history in this schema. If the user asks about past "
     "prices, trends or changes over time, output exactly: "
     "SELECT 'NO_HISTORY' AS note\n"
     "- If the question cannot be answered from this table, output exactly: "
     "SELECT 'UNANSWERABLE' AS note"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the LakVeggie assistant. Answer using ONLY the SQL result below. "
     "Never invent a price or a product. "
     "If the result is NO_HISTORY, explain that only current prices are stored, "
     "not past prices. "
     "If the result is UNANSWERABLE or empty, say the data is not available. "
     "Reply in the same language the user asked in (Sinhala or English). "
     "Product names stay in English even in a Sinhala reply. "
     "Format prices as 'Rs. 60.00'. Be brief."),
    ("human",
     "Question: {question}\n\nSQL executed:\n{sql}\n\nResult:\n{result}"),
])

sql_chain = SQL_PROMPT | llm | StrOutputParser()
answer_chain = ANSWER_PROMPT | llm | StrOutputParser()


# ----------------------------------------------------------------- FastAPI

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sql: str
    rows: str


app = FastAPI(title="LakVeggie AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # local dev only
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    try:
        db.run("SELECT 1")
        return {"status": "ok", "tables": ALLOWED_TABLES}
    except Exception as exc:                      # noqa: BLE001
        raise HTTPException(
            503, f"Database unreachable. Is the SSH tunnel up? {exc}"
        )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    schema = db.get_table_info()

    raw_sql = sql_chain.invoke({"schema": schema, "question": req.question})
    sql = sanitize_sql(raw_sql)
    log.info("SQL: %s", sql)

    try:
        rows = db.run(sql)
    except Exception as exc:                      # noqa: BLE001
        log.exception("query failed")
        raise HTTPException(500, f"Query failed: {exc}") from exc

    answer = answer_chain.invoke({
        "question": req.question,
        "sql": sql,
        "result": rows or "(no rows)",
    })

    return AskResponse(answer=answer, sql=sql, rows=str(rows))