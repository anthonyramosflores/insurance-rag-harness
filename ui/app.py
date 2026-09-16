"""
Streamlit Q&A interface for the Insurance RAG Agent -- routed through the
hallucination harness so flagged answers get retried, softened to a
fallback, or marked for human review instead of shown as confident fact.

Run with: streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from harness.harness import Route, run as run_harness

st.set_page_config(page_title="Insurance RAG Agent", page_icon="🏦", layout="wide")
st.title("Insurance RAG Agent")
st.caption("Ask questions about your policies, claims, and customers — answers are checked by the hallucination harness before you see them.")

with st.sidebar:
    st.header("Settings")
    source_filter = st.selectbox("Filter by data source", ["All", "policies", "claims", "customers"], index=0)
    n_results = st.slider("Retrieved chunks", 1, 10, 5)
    show_sources = st.toggle("Show retrieved sources", value=True)
    show_detail = st.toggle("Show harness detail (debug)", value=False)
    st.divider()
    st.markdown("**About**")
    st.markdown("Claude (Sonnet) generation + local sentence-transformers embeddings + ChromaDB, wrapped in a detection/routing harness.")

ROUTE_BADGES = {
    Route.PASS: "✅ passed harness checks",
    Route.RETRY: "🔁 regenerated after a groundedness flag",
    Route.FALLBACK: "⚠️ withheld — not well-grounded in the retrieved records",
    Route.HUMAN_REVIEW: "🧑‍⚖️ flagged for human review — model was inconsistent across samples",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about a policy, claim, or customer...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context, generating, and checking for hallucinations..."):
            try:
                result = run_harness(
                    user_input,
                    n_results=n_results,
                    source_filter=None if source_filter == "All" else source_filter,
                )

                st.markdown(result.answer)
                st.caption(ROUTE_BADGES.get(result.route, result.route.value))

                if show_sources and result.sources:
                    with st.expander("Retrieved sources", expanded=False):
                        for i, src in enumerate(result.sources, 1):
                            st.markdown(f"**[{i}]** `{src['metadata'].get('source', '?')}` — distance: `{src['distance']:.4f}`")
                            st.text(src["text"])

                if show_detail:
                    with st.expander("Harness detail", expanded=False):
                        st.json(result.detail)

                st.session_state.messages.append({"role": "assistant", "content": result.answer})

            except Exception as exc:
                error_msg = f"Error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                