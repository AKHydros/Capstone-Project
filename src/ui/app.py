from __future__ import annotations

from functools import lru_cache
from datetime import datetime
from pathlib import Path

import streamlit as st

from backend.config import load_config
from backend.services.bootstrap_service import BootstrapArtifacts, BootstrapService


@lru_cache(maxsize=1)
def build_service(force_rebuild_cache: bool = False, force_refresh_prompts: bool = False) -> BootstrapArtifacts:
    config = load_config()
    return BootstrapService(config).build(
        force_rebuild_cache=force_rebuild_cache,
        force_refresh_prompts=force_refresh_prompts,
    )


st.set_page_config(page_title="PMG Research Chatbot", layout="wide")
st.title("Research Data Dictionary Chatbot")
st.caption("Single source of truth: Excel data dictionary")

config = load_config()
data_dir = config.excel_source_path.parent if config.excel_source_path.parent.exists() else (Path.cwd() / "data")
uploads_dir = data_dir / "user_uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

if "force_rebuild_cache" not in st.session_state:
    st.session_state.force_rebuild_cache = False
if "force_refresh_prompts" not in st.session_state:
    st.session_state.force_refresh_prompts = False
if "pending_toast" not in st.session_state:
    st.session_state.pending_toast = None

try:
    with st.spinner("Loading search index..."):
        artifacts = build_service(
            force_rebuild_cache=st.session_state.force_rebuild_cache,
            force_refresh_prompts=st.session_state.force_refresh_prompts,
        )
        st.session_state.force_rebuild_cache = False
        st.session_state.force_refresh_prompts = False
except Exception as exc:  # noqa: BLE001
    st.error(f"Startup failed: {exc}")
    st.stop()

if st.session_state.pending_toast:
    st.toast(st.session_state.pending_toast)
    st.session_state.pending_toast = None

with st.sidebar:
    st.subheader("Maintenance")
    if st.button("Force Rebuild Cache", use_container_width=True):
        st.session_state.force_rebuild_cache = True
        st.session_state.pending_toast = "Cache rebuild requested."
        build_service.cache_clear()
        st.rerun()
    if st.button("Refresh Starter Prompts", use_container_width=True):
        st.session_state.force_refresh_prompts = True
        st.session_state.pending_toast = "Starter prompt refresh requested."
        build_service.cache_clear()
        st.rerun()

    st.subheader("Add Files")
    uploaded_files = st.file_uploader(
        "Upload additional .xlsx or .docx files",
        type=["xlsx", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Add Uploaded Files", use_container_width=True):
        saved_count = 0
        for file in uploaded_files or []:
            target = uploads_dir / file.name
            target.write_bytes(file.getbuffer())
            saved_count += 1
        if saved_count > 0:
            st.session_state.force_rebuild_cache = True
            st.session_state.force_refresh_prompts = True
            st.session_state.pending_toast = f"Added {saved_count} file(s). Cache and prompts refreshed."
            build_service.cache_clear()
            st.rerun()
        else:
            st.toast("No files selected.")

    st.subheader("Cache Status")
    built_at = (
        datetime.fromtimestamp(artifacts.cache_status.created_at_epoch).isoformat(timespec="seconds")
        if artifacts.cache_status.created_at_epoch
        else "N/A"
    )
    st.write(f"Startup State: **{artifacts.cache_status_at_startup}**")
    st.write(f"Current State: **{artifacts.cache_status.state}**")
    st.write(f"Action This Run: **{'rebuilt' if artifacts.cache_rebuilt else 'reused'}**")
    st.write(f"Last Build: `{built_at}`")
    st.write(f"Cache File: `{artifacts.cache_status.cache_file}`")

    st.subheader("Filters")
    survey_filter = st.selectbox("Survey", options=["All"] + artifacts.surveys, index=0)
    wave_filter = st.selectbox("Wave/Year", options=["All"] + artifacts.waves, index=0)

    st.subheader("Starter Questions")
    for idx, prompt in enumerate(artifacts.starter_prompts[:8], start=1):
        if st.button(f"{idx}. {prompt[:74]}", key=f"starter_{idx}"):
            st.session_state.prefill_prompt = prompt

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefill_prompt" not in st.session_state:
    st.session_state.prefill_prompt = ""

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about historical variables, labels, or mappings")
if not prompt and st.session_state.prefill_prompt:
    prompt = st.session_state.prefill_prompt
    st.session_state.prefill_prompt = ""

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        survey_name = None if survey_filter == "All" else survey_filter
        wave_year = None if wave_filter == "All" else wave_filter
        response = artifacts.chatbot_service.chat(prompt, survey_name=survey_name, wave_year=wave_year)

        st.markdown(response.answer)
        if response.ranked_results:
            st.markdown("### Retrieved Question Cards")
            for idx, record in enumerate(response.ranked_results, start=1):
                with st.expander(f"{idx}. {record.question_id} | {record.question_text[:90]}"):
                    st.write(f"**Question ID:** {record.question_id}")
                    st.write(f"**Question Text:** {record.question_text}")
                    st.write(f"**Survey:** {record.survey_name}")
                    st.write(f"**Wave/Year:** {record.wave_year}")
                    st.write(f"**Measurement Level:** {record.measurement_level}")
                    st.write(f"**Source File:** {record.source_file}")
                    if record.value_labels:
                        st.write("**Mapped Values (sample):**")
                        for v in record.value_labels[:10]:
                            st.write(f"- {v}")

        st.session_state.chat_history.append({"role": "assistant", "content": response.answer})
