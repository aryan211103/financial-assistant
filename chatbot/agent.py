"""
agent.py - Vertex AI ADK chatbot for the Financial Assistant.

Routes natural language questions to the right data source:
  - Postgres property and financial records
  - SEC EDGAR cached company financials
  - Mock press releases
and answers in natural language using Gemini on Vertex AI.

Test from the terminal (DATABASE_URL must be exported):
    python chatbot/agent.py
The Streamlit app imports ask() from this module.
"""

import os
import json
import asyncio
from pathlib import Path

import psycopg2

# Hosted: load the service account key from a secret into a temp file.
# Local: this is absent, so it falls back to your gcloud login.
_sa_key = os.environ.get("GCP_SA_KEY_JSON")
if _sa_key:
    with open("/tmp/gcp_key.json", "w") as f:
        f.write(_sa_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_key.json"

# Route ADK and genai to Vertex AI. Must be set before the agent is built.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"] = "financial-assistant-498905"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
APP_NAME = "financial_assistant"
MODEL = "gemini-2.5-flash"


# ---------- tools ----------
def query_properties(metro_area: str = "", property_type: str = "") -> str:
    """Look up the company's real estate properties and their financials.
    Use for questions about specific buildings: locations, square footage,
    revenue, net income, or expenses of properties the company owns.

    Args:
        metro_area: Optional city filter such as "Chicago". Empty means all cities.
        property_type: Optional type filter such as "Industrial", "Office",
            "Warehouse", "Retail", or "Mixed-Use". Empty means all types.

    Returns:
        A text list of matching properties with their financials.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        return "Database is not configured."
    conn = psycopg2.connect(url)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.address, p.metro_area, p.property_type, p.sq_footage,
                   f.revenue, f.net_income, f.expenses
            FROM properties p JOIN financials f ON p.property_id = f.property_id
            WHERE (%s = '' OR p.metro_area ILIKE %s)
              AND (%s = '' OR p.property_type ILIKE %s)
            ORDER BY p.property_id;
            """,
            (metro_area, metro_area, property_type, property_type),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return "No matching properties found."
    lines = []
    for a, metro, ptype, sqft, rev, ni, exp in rows:
        lines.append(
            f"{a} ({metro}, {ptype}, {sqft:,} sq ft): "
            f"revenue ${rev:,}, net income ${ni:,}, expenses ${exp:,}"
        )
    return "\n".join(lines)


def get_company_financials() -> str:
    """Get the company's overall financials from its latest SEC EDGAR filing.
    Use for company level questions like total revenue, net income, or
    operating expenses.

    Returns:
        A text summary of the company's revenue, net income, and operating expenses.
    """
    with open(DATA_DIR / "company_financials.json") as f:
        data = json.load(f)
    m = data["metrics"]
    return (
        f"{data['company']} ({data['ticker']}), fiscal year ending "
        f"{m['revenue']['fiscal_year_end']}:\n"
        f"Revenue: ${m['revenue']['value']:,}\n"
        f"Net income: ${m['net_income']['value']:,}\n"
        f"Operating expenses: ${m['operating_expenses']['value']:,}"
    )


def search_press_releases(category: str = "", keyword: str = "") -> str:
    """Search the company's recent press releases.
    Use for questions about announcements, acquisitions, expansions, earnings,
    leasing, or sustainability news.

    Args:
        category: Optional filter such as "acquisition", "expansion", "earnings",
            "leasing", or "sustainability". Empty means all categories.
        keyword: Optional keyword to match in the title or summary. Empty means
            no keyword filter.

    Returns:
        A text list of matching press releases.
    """
    with open(DATA_DIR / "press_releases.json") as f:
        data = json.load(f)
    out = []
    for pr in data["press_releases"]:
        if category and pr["category"].lower() != category.lower():
            continue
        if keyword and keyword.lower() not in (pr["title"] + pr["summary"]).lower():
            continue
        out.append(f"[{pr['date']}] {pr['title']} ({pr['category']}): {pr['summary']}")
    return "\n".join(out) if out else "No matching press releases found."


# ---------- agent ----------
agent = Agent(
    name="financial_assistant",
    model=MODEL,
    instruction=(
        "You are a financial assistant for Prologis, a real estate company. "
        "Answer questions using your tools: query_properties for specific "
        "buildings and their financials, get_company_financials for company "
        "wide revenue and income, and search_press_releases for announcements. "
        "Choose the right tool based on the question, then answer concisely in "
        "plain language. If a question does not fit any tool, answer briefly "
        "from general knowledge."
        "Format dollar amounts with a dollar sign and commas, for example $1,234,567. "
    ),
    tools=[query_properties, get_company_financials, search_press_releases],
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)


async def _ask_async(question: str, user_id: str = "user", session_id: str = "session") -> str:
    try:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception:
        pass  # session already exists, fine
    content = types.Content(role="user", parts=[types.Part(text=question)])
    final = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text or final
    return final


def ask(question: str, user_id: str = "user", session_id: str = "session") -> str:
    """Synchronous entry point used by the Streamlit app."""
    return asyncio.run(_ask_async(question, user_id, session_id))


if __name__ == "__main__":
    tests = [
        "Show industrial properties in Chicago with revenue details",
        "What was the company's net income last year?",
        "Did the company announce any acquisitions recently?",
    ]
    for q in tests:
        print("Q:", q)
        print("A:", ask(q))
        print("-" * 60)