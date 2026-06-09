"""
edgar_fetch.py
Pulls real financials for a public real estate company from SEC EDGAR
(the XBRL company facts API) and caches them to company_financials.json.
Caching means your app does not depend on EDGAR being reachable during the demo.

Install:  pip install requests
Run:      python edgar_fetch.py
"""

import json
import requests

# SEC requires a descriptive User-Agent with real contact info. Edit if you like.
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


def latest_annual_value(facts, tags):
    """Most recent 10-K annual USD value among the given candidate tags."""
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


def main():
    cik, name = get_cik(TICKER)
    print("Company:", name, "| CIK:", cik)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts = requests.get(url, headers=HEADERS, timeout=30).json()["facts"]

    out = {"company": name, "ticker": TICKER, "cik": cik, "metrics": {}}
    for metric, tags in METRIC_TAGS.items():
        found = latest_annual_value(facts, tags)
        if found:
            value = found["value"]
            # Some filers tag expenses as negative. Show a positive magnitude.
            if metric == "operating_expenses":
                value = abs(value)
            out["metrics"][metric] = {
                "value": value,
                "fiscal_year_end": found["end"],
                "source_tag": found["tag"],
            }
            print(f"{metric}: {value:,} (FY end {found['end']})")
        else:
            print(f"{metric}: NOT FOUND (tell me, we adjust the tag)")

    with open("company_financials.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved company_financials.json")


if __name__ == "__main__":
    main()
