-- schema.sql
-- Database schema for the Financial Assistant property data source.
-- This mirrors the tables created programmatically in db_setup.py.
-- db_setup.py also seeds 20 sample property records and their financials.

-- Properties: one row per building the company owns.
CREATE TABLE IF NOT EXISTS properties (
    property_id   INTEGER PRIMARY KEY,
    address       TEXT    NOT NULL,
    metro_area    TEXT    NOT NULL,
    sq_footage    INTEGER NOT NULL,
    property_type TEXT    NOT NULL
);

-- Financials: one row per property, linked by property_id.
CREATE TABLE IF NOT EXISTS financials (
    property_id INTEGER REFERENCES properties(property_id),
    revenue     BIGINT,
    net_income  BIGINT,
    expenses    BIGINT
);

-- Example query used by the app and the chatbot:
-- industrial properties in Chicago with their financials.
-- SELECT p.address, p.metro_area, p.property_type, p.sq_footage,
--        f.revenue, f.net_income, f.expenses
-- FROM properties p
-- JOIN financials f ON p.property_id = f.property_id
-- WHERE p.metro_area = 'Chicago' AND p.property_type = 'Industrial';