"""
Load raw CSV files from disk into pandas DataFrames.
"""

import os
import pandas as pd

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

EXPECTED_FILES = {
    "policies": "policies.csv",
    "claims": "claims.csv",
    "customers": "customers.csv",
}


def load_csv(name: str, data_dir: str = LOCAL_DATA_DIR) -> pd.DataFrame:
    filename = EXPECTED_FILES.get(name)
    if not filename:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(EXPECTED_FILES)}")
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run ingestion/mock_data.py first.")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_all(data_dir: str = LOCAL_DATA_DIR) -> dict[str, pd.DataFrame]:
    return {name: load_csv(name, data_dir) for name in EXPECTED_FILES}