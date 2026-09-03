"""
Pandas transformations: clean, enrich, and normalize raw insurance DataFrames
"""

import pandas as pd

def clean_policies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ("premium_usd", "coverage_amount_usd"):  
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df =df.dropna(subset=["policy_id", "customer_id"])
    return df

def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")

    for col in ("amount_claimed_usd", "amount_paid_usd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["payout_ratio"] = (df["amount_paid_usd"] / df["amount_claimed_usd"]).clip(0, 1)
    return df


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    return df


def run_all_transforms(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "policies": clean_policies(raw["policies"]),
        "claims": clean_claims(raw["claims"]),
        "customers": clean_customers(raw["customers"]),
    }

