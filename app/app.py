"""
app.py - Financial Assistant for a Real Estate Company (Prologis demo)

Streamlit front end that ties together:
  - Postgres (property and financial records)
  - SEC EDGAR cached financials
  - Mock press releases
  - Two SageMaker endpoints (regression and classification)

The Chatbot tab is filled in later, once the Vertex AI ADK agent exists.

Run from the repo root:
    pip install streamlit psycopg2-binary boto3
    export DATABASE_URL="your_neon_connection_string"
    streamlit run app/app.py
"""

import os
import json
from pathlib import Path

import boto3
import psycopg2
import streamlit as st

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "chatbot"))
from agent import ask

# ---------- config ----------
REGION = "us-east-1"
REG_ENDPOINT = "housing-regression"
CLF_ENDPOINT = "bank-subscription"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Real Estate Financial Assistant", layout="wide")


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
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = rt.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(resp["Body"].read().decode())

def bedrock_summarize(text):
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    prompt = (
        "Summarize these company press releases into one short paragraph, "
        "highlighting key themes like acquisitions, expansions, and earnings:\n\n" + text
    )
    resp = client.converse(
        modelId="us.amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 400, "temperature": 0.3},
    )
    return resp["output"]["message"]["content"][0]["text"]

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


# ---------- UI ----------
st.title("Real Estate Financial Assistant")
st.caption("Prologis demo. Sources: Postgres, SEC EDGAR, press releases, and SageMaker models.")

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
            answer = ask(prompt)
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
        rows = run_query(sql, (metro, metro, ptype, ptype))
        st.dataframe(rows, use_container_width=True)

# --- Press Releases ---
with tab_news:
    st.subheader("Recent press releases")
    pr = load_json("press_releases.json")
    if st.button("Extract key insights (AWS Comprehend)"):
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
        out = invoke(REG_ENDPOINT, payload)
        val = out["predictions"][0] * 100000
        st.success(f"Predicted median house value: ${val:,.0f}")

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
        out = invoke(CLF_ENDPOINT, {"instances": [instance]})
        pred = out["predictions"][0]
        prob = pred["probability"] * 100
        if pred["label"] == 1:
            st.success(f"Likely to subscribe ({prob:.1f}% probability)")
        else:
            st.warning(f"Unlikely to subscribe ({prob:.1f}% probability)")