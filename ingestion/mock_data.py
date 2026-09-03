"""
Generate mock insurance CSV data for development and testing.
"""

import csv
import os
import random
from datetime import date, timedelta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

POLICY_TYPES = ["Auto", "Home", "Life", "Health", "Commercial"]
STATUSES = ["Active", "Lapsed", "Cancelled", "Pending"]
STATES = ["CA", "TX", "NY", "FL", "IL", "WA", "CO", "AZ"]

CLAIMS_DESCRIPTIONS = [
    "Vehicle collision on highway, rear-end damage reported.",
    "Water damage from burst pipe in kitchen.",
    "Medical emergency hospitalization claim.",
    "Theft of personal property from residence.",
    "Hail damage to roof and siding.",
]


def random_date(start_year: int = 2020, end_year: int = 2025) -> str:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = end - start
    return str(start + timedelta(days=random.randint(0, delta.days)))


def generate_policies(n: int = 200) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "policy_id": f"POL-{i:05d}",
            "customer_id": f"CUST-{random.randint(1, 500):05d}",
            "policy_type": random.choice(POLICY_TYPES),
            "status": random.choice(STATUSES),
            "start_date": random_date(2020, 2022),
            "end_date": random_date(2023, 2025),
            "premium_usd": round(random.uniform(300, 5000), 2),
            "coverage_amount_usd": round(random.uniform(10_000, 1_000_000), 2),
            "state": random.choice(STATES),
        })
    return rows


def generate_claims(n: int = 150) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "claim_id": f"CLM-{i:05d}",
            "policy_id": f"POL-{random.randint(1, 200):05d}",
            "claim_date": random_date(2022, 2025),
            "status": random.choice(["Open", "Closed", "Under Review", "Denied"]),
            "amount_claimed_usd": round(random.uniform(500, 150_000), 2),
            "amount_paid_usd": round(random.uniform(0, 100_000), 2),
            "description": random.choice(CLAIMS_DESCRIPTIONS),
        })
    return rows


def generate_customers(n: int = 500) -> list[dict]:
    first_names = ["Alice", "Bob", "Carol", "David", "Eva", "Frank"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "customer_id": f"CUST-{i:05d}",
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "state": random.choice(STATES),
            "age": random.randint(18, 80),
            "risk_score": round(random.uniform(0.1, 1.0), 2),
        })
    return rows


def write_csv(rows: list[dict], filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")
    return path


if __name__ == "__main__":
    write_csv(generate_policies(), "policies.csv")
    write_csv(generate_claims(), "claims.csv")
    write_csv(generate_customers(), "customers.csv")