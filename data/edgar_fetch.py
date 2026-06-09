"""
edgar_fetch.py
Pulls real financials for a public real estate company from SEC EDGAR
(the XBRL company facts API) and caches them to company_financials.json.

Now extracts BOTH:
  - annual figures from the latest 10-K (fiscal year)
  - the most recent single quarter from 10-Q filings

Caching means the app does not depend on EDGAR being reachable at demo time.

Install:  pip install requests
Run:      python edgar_fetch.py
"""

import json
from datetime import datetime

import requests

# SEC requires a descriptive User-Agent with real contact info.
HEADERS = {"User-Agent": "Aryan Hirlekar hirlekar.a@northeastern.edu"}
TICKER = "PLD"   # Prologis, a large industrial REIT

# XBRL tags to try, in order, for each metric (handles tag variation)
METRIC_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_expenses": ["OperatingExpenses", "CostsAndExpenses"],
}


def get_cik(ticker):
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10), row["title"]
    raise ValueError(f"Ticker {ticker} not found in SEC list")


def _days(start, end):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    return (d1 - d0).days


def latest_annual_value(facts, tags):
    """Most recent 10-K annual USD value among the candidate tags."""
    best = None
    for tag in tags:
        node = facts.get("us-gaap", {}).get(tag)
        if not node:
            continue
        for v in node.get("units", {}).get("USD", []):
            if v.get("form") == "10-K" and v.get("fp") == "FY" and v.get("end"):
                if best is None or v["end"] > best["end"]:
                    best = {"value": v["val"], "end": v["end"], "tag": tag}
    return best


def latest_quarter_value(facts, tags):
    """Most recent single-quarter (about 3 months) value from 10-Q filings.
    YTD periods (6 or 9 months) are skipped by the duration filter."""
    best = None
    for tag in tags:
        node = facts.get("us-gaap", {}).get(tag)
        if not node:
            continue
        for v in node.get("units", {}).get("USD", []):
            if v.get("form") != "10-Q" or not v.get("start") or not v.get("end"):
                continue
            duration = _days(v["start"], v["end"])
            if 80 <= duration <= 100:  # one quarter, not a year-to-date sum
                if best is None or v["end"] > best["end"]:
                    best = {
                        "value": v["val"],
                        "end": v["end"],
                        "tag": tag,
                        "fp": v.get("fp"),
                    }
    return best


def main():
    cik, name = get_cik(TICKER)
    print("Company:", name, "| CIK:", cik)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts = requests.get(url, headers=HEADERS, timeout=30).json()["facts"]

    out = {"company": name, "ticker": TICKER, "cik": cik, "metrics": {}, "quarterly": {}}

    for metric, tags in METRIC_TAGS.items():
        annual = latest_annual_value(facts, tags)
        if annual:
            val = abs(annual["value"]) if metric == "operating_expenses" else annual["value"]
            out["metrics"][metric] = {
                "value": val,
                "fiscal_year_end": annual["end"],
                "source_tag": annual["tag"],
            }
            print(f"[annual]  {metric}: {val:,} (FY end {annual['end']})")
        else:
            print(f"[annual]  {metric}: NOT FOUND")

        quarter = latest_quarter_value(facts, tags)
        if quarter:
            val = abs(quarter["value"]) if metric == "operating_expenses" else quarter["value"]
            out["quarterly"][metric] = {
                "value": val,
                "quarter_end": quarter["end"],
                "fiscal_period": quarter["fp"],
                "source_tag": quarter["tag"],
            }
            print(f"[quarter] {metric}: {val:,} (quarter end {quarter['end']}, {quarter['fp']})")
        else:
            print(f"[quarter] {metric}: NOT FOUND (only year-to-date reported, that is fine)")

    with open("company_financials.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved company_financials.json")


if __name__ == "__main__":
    main()