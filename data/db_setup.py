"""
db_setup.py
Creates the Properties and Financials tables in your hosted Postgres and
seeds 20 property records plus their financials.

Set your connection string first (paste the full URI from Supabase / Neon):
    export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
Install:  pip install psycopg2-binary
Run:      python db_setup.py
"""

import os
import psycopg2

DB_URL = os.environ["DATABASE_URL"]

# (property_id, address, metro_area, sq_footage, property_type)
PROPERTIES = [
    (1,  "1200 W Fulton St",     "Chicago",       450000, "Industrial"),
    (2,  "850 S Cicero Ave",     "Chicago",       620000, "Industrial"),
    (3,  "401 N Michigan Ave",   "Chicago",       310000, "Office"),
    (4,  "2300 Logistics Pkwy",  "Chicago",       780000, "Warehouse"),
    (5,  "500 Industrial Blvd",  "Dallas",        540000, "Industrial"),
    (6,  "1700 Commerce St",     "Dallas",        290000, "Office"),
    (7,  "9000 Distribution Dr", "Atlanta",       710000, "Warehouse"),
    (8,  "120 Peachtree St",     "Atlanta",       260000, "Office"),
    (9,  "4500 Logistics Way",   "Los Angeles",   830000, "Industrial"),
    (10, "700 Harbor Blvd",      "Los Angeles",   410000, "Industrial"),
    (11, "55 Retail Plaza",      "Los Angeles",   180000, "Retail"),
    (12, "300 Tech Center Dr",   "Austin",        350000, "Office"),
    (13, "1450 Freight Ln",      "Austin",        600000, "Warehouse"),
    (14, "88 Market St",         "San Francisco", 220000, "Office"),
    (15, "2100 Bayfront Ave",    "San Francisco", 150000, "Retail"),
    (16, "640 Cargo Rd",         "Newark",        690000, "Industrial"),
    (17, "12 Liberty Ave",       "Newark",        240000, "Mixed-Use"),
    (18, "900 Seaport Blvd",     "Seattle",       520000, "Warehouse"),
    (19, "75 Pine St",           "Seattle",       200000, "Office"),
    (20, "3300 Gateway Dr",      "Miami",         470000, "Industrial"),
]


def make_financials():
    """Financials roughly scale with square footage. Values in USD."""
    rows = []
    for pid, _, _, sqft, _ in PROPERTIES:
        revenue = sqft * 18
        expenses = int(revenue * 0.55)
        net_income = revenue - expenses
        rows.append((pid, revenue, net_income, expenses))
    return rows


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS financials;")
    cur.execute("DROP TABLE IF EXISTS properties;")

    cur.execute("""
        CREATE TABLE properties (
            property_id   INTEGER PRIMARY KEY,
            address       TEXT,
            metro_area    TEXT,
            sq_footage    INTEGER,
            property_type TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE financials (
            property_id INTEGER REFERENCES properties(property_id),
            revenue     BIGINT,
            net_income  BIGINT,
            expenses    BIGINT
        );
    """)

    cur.executemany("INSERT INTO properties VALUES (%s, %s, %s, %s, %s);", PROPERTIES)
    cur.executemany("INSERT INTO financials VALUES (%s, %s, %s, %s);", make_financials())
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM properties;")
    print("Properties rows:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM financials;")
    print("Financials rows:", cur.fetchone()[0])

    # Sanity check the kind of query your chatbot will run
    cur.execute("""
        SELECT p.address, p.metro_area, f.revenue
        FROM properties p JOIN financials f ON p.property_id = f.property_id
        WHERE p.property_type = 'Industrial' AND p.metro_area = 'Chicago';
    """)
    print("Industrial in Chicago:", cur.fetchall())

    cur.close()
    conn.close()
    print("Database ready.")


if __name__ == "__main__":
    main()
