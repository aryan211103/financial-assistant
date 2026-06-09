"""
app.py - Financial Assistant for a Real Estate Company (Prologis demo)

Streamlit front end that ties together:
  - Postgres (property and financial records)
  - SEC EDGAR cached financials (annual and latest quarter)
  - Mock press releases (with text-derived insight extraction)
  - Two SageMaker endpoints (regression and classification)
  - A Vertex AI ADK chatbot

Run from the repo root:
    pip install streamlit psycopg2-binary boto3
    export DATABASE_URL="your_neon_connection_string"
    streamlit run app/app.py
"""

import os
import json
import sys
import time
from pathlib import Path

import boto3
import psycopg2
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "chatbot"))
from agent import ask

# ---------- config ----------
REGION = "us-east-1"
REG_ENDPOINT = "housing-regression"
CLF_ENDPOINT = "bank-subscription"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Real Estate Financial Assistant", layout="wide")


# ---------- styling ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, div, span, label, input, select, textarea {
    font-family: 'IBM Plex Sans', sans-serif;
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: #1A2233; letter-spacing: -0.01em; }

/* hero band */
.hero {
    background: linear-gradient(135deg, #1F3A5F 0%, #2A4D73 100%);
    color: #FFFFFF; padding: 26px 30px; border-radius: 14px; margin-bottom: 22px;
}
.hero h1 { color: #FFFFFF; margin: 0 0 6px 0; font-size: 28px; }
.hero p { color: #C7D3E2; margin: 0; font-size: 15px; }
.hero .pill {
    display: inline-block; margin-top: 14px; padding: 5px 13px; border-radius: 999px;
    background: rgba(255,255,255,0.12); color: #DCE6F2; font-size: 12px; letter-spacing: 0.04em;
}

/* metric cards as ledger tiles */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E6E9EF; border-radius: 12px;
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
[data-testid="stMetricLabel"] { color: #6B7280; font-size: 13px; }
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: #1A2233;
    font-variant-numeric: tabular-nums; font-size: 1.45rem;
}

/* buttons */
.stButton > button {
    background: #1F3A5F; color: #FFFFFF; border: none; border-radius: 10px;
    padding: 8px 18px; font-weight: 600; font-family: 'IBM Plex Sans', sans-serif;
    transition: background 0.15s ease;
}
.stButton > button:hover { background: #16314F; color: #FFFFFF; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 15px;
    padding: 8px 14px;
}
.stTabs [aria-selected="true"] { color: #1F3A5F; }

/* sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E6E9EF; }

/* hide default chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- helpers ----------
def get_db_url():
    url = os.environ.get("DATABASE_URL")
    if not url:
        try:
            url = st.secrets["DATABASE_URL"]
        except Exception:
            url = None
    return url


@st.cache_data
def load_json(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)


def run_query(sql, params=None):
    conn = psycopg2.connect(get_db_url())
    try:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()


def invoke(endpoint, payload):
    """Call a SageMaker endpoint and return (parsed_response, latency_ms)."""
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    start = time.time()
    resp = rt.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    latency_ms = (time.time() - start) * 1000
    out = json.loads(resp["Body"].read().decode())
    return out, latency_ms


def comprehend_extract(text):
    client = boto3.client("comprehend", region_name="us-east-1")
    snippet = text[:4500]  # sync API limit is 5000 bytes
    sentiment = client.detect_sentiment(Text=snippet, LanguageCode="en")
    phrases = client.detect_key_phrases(Text=snippet, LanguageCode="en")
    seen = []
    for k in sorted(phrases["KeyPhrases"], key=lambda x: x["Score"], reverse=True):
        t = k["Text"].strip()
        if t.lower() not in [s.lower() for s in seen]:
            seen.append(t)
    return {"sentiment": sentiment["Sentiment"].title(), "key_phrases": seen[:10]}


def extract_insight(text):
    """Derive an insight type from the press release text (not the stored label)."""
    t = text.lower()
    if any(k in t for k in ["solar", "sustainab", "carbon", "emission"]):
        return "Sustainability"
    if any(k in t for k in ["quarter", "full year", "earnings", "results"]):
        return "Quarterly update"
    if any(k in t for k in ["lease", "tenant", "retailer"]):
        return "Leasing"
    if any(k in t for k in ["acquir", "acquisition", "purchase"]):
        return "Acquisition"
    if any(k in t for k in ["expand", "construction", "breaks ground", "distribution center"]):
        return "Expansion"
    return "Other"


# ---------- debugging / observability helpers ----------
def check_database():
    try:
        run_query("SELECT 1;")
        return True, "connected"
    except Exception as e:
        return False, str(e)


def check_endpoint(name):
    try:
        sm = boto3.client("sagemaker", region_name=REGION)
        status = sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
        return status == "InService", status
    except Exception as e:
        return False, str(e)


def check_chatbot():
    try:
        import agent  # noqa: F401
        vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
        return True, f"agent loaded (Vertex={vertex})"
    except Exception as e:
        return False, str(e)


# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "Financial assistant for Prologis (PLD). It combines Postgres property "
        "data, SEC EDGAR financials, press releases, two SageMaker ML models, "
        "and a Vertex AI ADK chatbot."
    )
    st.divider()
    st.subheader("Data sources and services")
    st.markdown(
        "- Postgres (Neon): property records\n"
        "- SEC EDGAR: company financials\n"
        "- Press releases: JSON, AWS Comprehend\n"
        "- SageMaker: regression and classification\n"
        "- Vertex AI ADK: chatbot"
    )
    st.divider()
    st.subheader("System status")
    st.caption("Live check of the backend services.")
    if st.button("Run health check"):
        ok, detail = check_database()
        (st.success if ok else st.error)(f"Database: {detail}")

        ok, detail = check_endpoint(REG_ENDPOINT)
        (st.success if ok else st.error)(f"Regression endpoint: {detail}")

        ok, detail = check_endpoint(CLF_ENDPOINT)
        (st.success if ok else st.error)(f"Classification endpoint: {detail}")

        ok, detail = check_chatbot()
        (st.success if ok else st.error)(f"Chatbot: {detail}")


# ---------- hero ----------
st.markdown(
    '<div class="hero">'
    '<h1>Real Estate Financial Assistant</h1>'
    '<p>Query company financials, properties, and news, and run ML predictions for Prologis (PLD).</p>'
    '<span class="pill">Postgres &nbsp;·&nbsp; SEC EDGAR &nbsp;·&nbsp; Press Releases &nbsp;·&nbsp; SageMaker &nbsp;·&nbsp; Vertex AI</span>'
    '</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_data, tab_news, tab_ml = st.tabs(
    ["Chatbot", "Properties and Financials", "Press Releases", "ML Predictions"]
)

# --- Chatbot ---
with tab_chat:
    st.subheader("Ask the assistant")
    st.caption("Ask about properties, company financials, or press releases.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", "\\$"))
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Your question",
                               placeholder="e.g. Show industrial properties in Chicago")
        submitted = st.form_submit_button("Ask")
    if submitted and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            try:
                answer = ask(prompt)
            except Exception as e:
                answer = f"The assistant hit an error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# --- Properties and Financials ---
with tab_data:
    st.subheader("Company financials (SEC EDGAR)")
    fin = load_json("company_financials.json")
    m = fin["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", f"${m['revenue']['value']:,}")
    c2.metric("Net income", f"${m['net_income']['value']:,}")
    c3.metric("Operating expenses", f"${m['operating_expenses']['value']:,}")
    st.caption(f"{fin['company']} ({fin['ticker']}), FY end {m['revenue']['fiscal_year_end']}")

    q = fin.get("quarterly")
    if q:
        end = (q.get("net_income") or next(iter(q.values()))).get("quarter_end")
        st.markdown(f"**Most recent quarter (ending {end})**")
        qc1, qc2, qc3 = st.columns(3)
        if "revenue" in q:
            qc1.metric("Revenue", f"${q['revenue']['value']:,}")
        if "net_income" in q:
            qc2.metric("Net income", f"${q['net_income']['value']:,}")
        if "operating_expenses" in q:
            qc3.metric("Operating expenses", f"${q['operating_expenses']['value']:,}")

    st.divider()
    st.subheader("Property records (Postgres)")
    if not get_db_url():
        st.error("DATABASE_URL is not set. Export it before running the app.")
    else:
        metros = ["All", "Chicago", "Dallas", "Atlanta", "Los Angeles",
                  "Austin", "San Francisco", "Newark", "Seattle", "Miami"]
        types = ["All", "Industrial", "Office", "Warehouse", "Retail", "Mixed-Use"]
        f1, f2 = st.columns(2)
        metro = f1.selectbox("Metro area", metros)
        ptype = f2.selectbox("Property type", types)

        sql = """
            SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
                   p.property_type, f.revenue, f.net_income, f.expenses
            FROM properties p JOIN financials f ON p.property_id = f.property_id
            WHERE (%s = 'All' OR p.metro_area = %s)
              AND (%s = 'All' OR p.property_type = %s)
            ORDER BY p.property_id;
        """
        try:
            rows = run_query(sql, (metro, metro, ptype, ptype))
            st.caption(f"{len(rows)} matching properties")
            st.dataframe(rows, use_container_width=True)
        except Exception as e:
            st.error(f"Database query failed: {e}")

# --- Press Releases ---
with tab_news:
    st.subheader("Recent press releases")
    pr = load_json("press_releases.json")

    st.markdown("**Extracted insights**")
    insights = {}
    for p in pr["press_releases"]:
        kind = extract_insight(p["title"] + " " + p["summary"])
        insights.setdefault(kind, []).append(p["title"])
    st.caption(", ".join(f"{k}: {len(v)}" for k, v in sorted(insights.items())))

    b1, b2 = st.columns(2)
    if b1.button("Extract key insights (AWS Comprehend)"):
        joined = "\n".join(
            f"{p['date']} {p['title']}: {p['summary']}" for p in pr["press_releases"]
        )
        with st.spinner("Analyzing with AWS Comprehend..."):
            try:
                result = comprehend_extract(joined)
                st.write(f"**Overall sentiment:** {result['sentiment']}")
                st.write("**Key phrases:** " + ", ".join(result["key_phrases"]))
            except Exception as e:
                st.error(f"Comprehend error: {e}")

    if b2.button("Summarize all recent news"):
        joined = "\n".join(f"{p['title']}: {p['summary']}" for p in pr["press_releases"])
        with st.spinner("Summarizing..."):
            try:
                st.info(ask("Summarize these press releases in 3 sentences: " + joined))
            except Exception as e:
                st.error(f"Summary error: {e}")

    cats = ["All"] + sorted({p["category"] for p in pr["press_releases"]})
    pick = st.selectbox("Category", cats)
    for item in pr["press_releases"]:
        if pick == "All" or item["category"] == pick:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.caption(f"{item['date']} | {item['category']}")
                st.write(item["summary"])

# --- ML Predictions ---
with tab_ml:
    st.subheader("Regression: predict median house value")
    st.caption("California Housing Random Forest on a SageMaker endpoint.")
    r1, r2, r3, r4 = st.columns(4)
    med_inc = r1.number_input("Median income (10k)", value=8.33)
    house_age = r2.number_input("House age", value=41.0)
    ave_rooms = r3.number_input("Avg rooms", value=6.98)
    ave_bed = r4.number_input("Avg bedrooms", value=1.02)
    r5, r6, r7, r8 = st.columns(4)
    pop = r5.number_input("Population", value=322.0)
    ave_occ = r6.number_input("Avg occupancy", value=2.55)
    lat = r7.number_input("Latitude", value=37.88)
    lon = r8.number_input("Longitude", value=-122.23)
    if st.button("Predict house value"):
        payload = {"instances": [[med_inc, house_age, ave_rooms, ave_bed,
                                   pop, ave_occ, lat, lon]]}
        try:
            out, ms = invoke(REG_ENDPOINT, payload)
            val = out["predictions"][0] * 100000
            st.metric("Predicted median house value", f"${val:,.0f}")
            st.caption(f"SageMaker endpoint responded in {ms:.0f} ms")
            with st.expander("Raw endpoint response"):
                st.json(out)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()
    st.subheader("Classification: predict customer subscription")
    st.caption("Bank Marketing Logistic Regression on a SageMaker endpoint.")
    cc1, cc2, cc3 = st.columns(3)
    age = cc1.number_input("Age", value=35)
    balance = cc2.number_input("Balance", value=1200)
    duration = cc3.number_input("Call duration (sec)", value=220)
    cc4, cc5, cc6 = st.columns(3)
    job = cc4.selectbox("Job", ["admin.", "blue-collar", "technician", "management",
                                "services", "retired", "student", "unemployed",
                                "entrepreneur", "housemaid", "self-employed", "unknown"])
    marital = cc5.selectbox("Marital", ["married", "single", "divorced"])
    education = cc6.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    cc7, cc8, cc9 = st.columns(3)
    housing = cc7.selectbox("Housing loan", ["yes", "no"])
    loan = cc8.selectbox("Personal loan", ["yes", "no"])
    contact = cc9.selectbox("Contact", ["cellular", "telephone", "unknown"])
    cc10, cc11, cc12 = st.columns(3)
    month = cc10.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun",
                                     "jul", "aug", "sep", "oct", "nov", "dec"])
    poutcome = cc11.selectbox("Previous outcome", ["unknown", "failure", "other", "success"])
    default = cc12.selectbox("Credit default", ["no", "yes"])

    if st.button("Predict subscription"):
        instance = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing,
            "loan": loan, "contact": contact, "day_of_week": 5, "month": month,
            "duration": duration, "campaign": 2, "pdays": -1, "previous": 0,
            "poutcome": poutcome,
        }
        try:
            out, ms = invoke(CLF_ENDPOINT, {"instances": [instance]})
            pred = out["predictions"][0]
            prob = pred["probability"] * 100
            label = "Likely to subscribe" if pred["label"] == 1 else "Unlikely to subscribe"
            st.metric(label, f"{prob:.1f}% probability")
            st.caption(f"SageMaker endpoint responded in {ms:.0f} ms")
            with st.expander("Raw endpoint response"):
                st.json(out)
        except Exception as e:
            st.error(f"Prediction failed: {e}")