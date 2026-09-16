"""
Eval dashboard: hallucination rate, catch rate, false-positive rate, and a
browsable log of every harness run.

Run with: streamlit run evaluation/dashboard.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

from evaluation.logging_db import fetch_all_runs

st.set_page_config(page_title="Hallucination Harness Dashboard", page_icon="🛡️", layout="wide")
st.title("Hallucination Harness — Eval Dashboard")

runs = fetch_all_runs()

if not runs:
    st.info("No harness runs logged yet. Ask a few questions through the UI or run harness/eval_set.py first.")
    st.stop()

df = pd.DataFrame(runs)
df["timestamp"] = pd.to_datetime(df["timestamp"])

total = len(df)
flagged = int(df["flagged"].sum())
hallucination_rate = flagged / total if total else 0.0

labeled = df[df["label"].notna()]
hallucination_prone = labeled[labeled["label"] == "hallucination_prone"]
clean = labeled[labeled["label"] == "clean"]

catch_rate = (hallucination_prone["flagged"] == 1).mean() if len(hallucination_prone) else None
false_positive_rate = (clean["flagged"] == 1).mean() if len(clean) else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total runs", total)
col2.metric("Flag rate (all runs)", f"{hallucination_rate:.0%}")
col3.metric("Catch rate (recall)", f"{catch_rate:.0%}" if catch_rate is not None else "—")
col4.metric("False positive rate", f"{false_positive_rate:.0%}" if false_positive_rate is not None else "—")

st.caption(
    "Catch rate / false positive rate only populate for runs from harness/eval_set.py "
    "(they carry a ground-truth label) — ad-hoc UI queries have no label."
)

st.subheader("Route breakdown")
route_counts = df["route"].value_counts().rename_axis("route").reset_index(name="count")
st.bar_chart(route_counts.set_index("route"))

st.subheader("Run log")
display_cols = ["timestamp", "query", "route", "groundedness_score", "confidence_flagged", "label"]
st.dataframe(df[display_cols], use_container_width=True)

with st.expander("Inspect a single run"):
    run_id = st.selectbox("Run", df["id"].tolist())
    row = df[df["id"] == run_id].iloc[0]
    st.markdown(f"**Query:** {row['query']}")
    st.markdown(f"**Answer:** {row['answer']}")
    st.markdown(f"**Route:** `{row['route']}`")
    st.json(row["detail_json"])
    