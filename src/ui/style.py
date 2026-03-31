from __future__ import annotations

import streamlit as st


def apply_pmg_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1180px;
            padding-top: 0.9rem;
            padding-bottom: 1.5rem;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 0.9rem;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-bottom: 0.3rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            margin-bottom: 0.35rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 0.55rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 0.55rem;
            padding: 0.25rem 0.4rem;
        }

        [data-testid="stChatInput"] {
            margin-top: 0.55rem;
        }

        [data-testid="stMarkdownContainer"] h6 {
            margin-bottom: 0.2rem;
        }

        [data-testid="stCaptionContainer"] p {
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
