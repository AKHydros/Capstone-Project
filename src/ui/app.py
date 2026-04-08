from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from backend.api.client import ApiClient
from backend.config import load_config
from backend.services.bootstrap_service import BootstrapArtifacts, BootstrapService
from ui.style import apply_pmg_theme


@lru_cache(maxsize=1)
def build_runtime(force_rebuild_cache: bool = False, force_refresh_prompts: bool = False) -> BootstrapArtifacts:
    config = load_config()
    return BootstrapService(config).build(
        force_rebuild_cache=force_rebuild_cache,
        force_refresh_prompts=force_refresh_prompts,
    )


def _init_state() -> None:
    if "force_rebuild_cache" not in st.session_state:
        st.session_state.force_rebuild_cache = False
    if "force_refresh_prompts" not in st.session_state:
        st.session_state.force_refresh_prompts = False
    if "pending_toast" not in st.session_state:
        st.session_state.pending_toast = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "prefill_prompt" not in st.session_state:
        st.session_state.prefill_prompt = ""
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "router_mode" not in st.session_state:
        st.session_state.router_mode = "hybrid"
    if "consent_granted" not in st.session_state:
        st.session_state.consent_granted = False
    if "trace_lookup_result" not in st.session_state:
        st.session_state.trace_lookup_result = None
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "chatgpt"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = load_config().openai_chat_model
    if "input_method" not in st.session_state:
        st.session_state.input_method = "Document"
    if "translation_method" not in st.session_state:
        st.session_state.translation_method = "Built-in Dictionary"
    if "inference_max_length_enabled" not in st.session_state:
        st.session_state.inference_max_length_enabled = False
    if "inference_top_p_enabled" not in st.session_state:
        st.session_state.inference_top_p_enabled = False
    if "inference_temperature_enabled" not in st.session_state:
        st.session_state.inference_temperature_enabled = False
    if "inference_max_length" not in st.session_state:
        st.session_state.inference_max_length = 512
    if "inference_top_p" not in st.session_state:
        st.session_state.inference_top_p = 1.0
    if "inference_temperature" not in st.session_state:
        st.session_state.inference_temperature = 0.1
    if "show_admin_dashboard" not in st.session_state:
        st.session_state.show_admin_dashboard = False
    if "session_consent_applied" not in st.session_state:
        st.session_state.session_consent_applied = False


def _save_uploaded_files(uploads_dir: Path, uploaded_files: list[object] | None) -> int:
    saved_count = 0
    for file in uploaded_files or []:
        target = uploads_dir / file.name
        target.write_bytes(file.getbuffer())
        saved_count += 1
    return saved_count


def _cache_status_text(artifacts: BootstrapArtifacts) -> tuple[str, str, str, str]:
    built_at = (
        datetime.fromtimestamp(artifacts.cache_status.created_at_epoch, tz=timezone.utc).isoformat(timespec="seconds")
        if artifacts.cache_status.created_at_epoch
        else "N/A"
    )
    return (
        artifacts.cache_status_at_startup,
        artifacts.cache_status.state,
        "rebuilt" if artifacts.cache_rebuilt else "reused",
        built_at,
    )


def _request_cache_action(*, rebuild: bool, refresh_prompts: bool, toast_message: str) -> None:
    st.session_state.force_rebuild_cache = rebuild
    st.session_state.force_refresh_prompts = refresh_prompts
    st.session_state.pending_toast = toast_message
    build_runtime.cache_clear()
    st.rerun()


def _ensure_chat_session_enabled(client: ApiClient) -> None:
    if st.session_state.session_consent_applied:
        return
    try:
        client.consent_record(
            session_id=st.session_state.session_id,
            user_consent=True,
            locale=st.session_state.language,
        )
        st.session_state.consent_granted = True
        st.session_state.session_consent_applied = True
    except Exception:  # noqa: BLE001
        # Best effort. If this fails, the router may still enforce consent.
        st.session_state.pending_toast = "Could not auto-enable chat session consent."


def _combine_starter_prompts(starter_prompts: list[str], top_questions: list[str]) -> list[str]:
    combined: list[str] = []
    seen: set[str] = set()
    for prompt in [*top_questions, *starter_prompts]:
        normalized = (prompt or "").strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        combined.append(normalized)
    return combined


def _chat_history_text(history: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines) if lines else "No chat messages yet."


def _render_admin_dashboard(client: ApiClient) -> None:
    st.markdown("### Admin & Monitoring Dashboard")

    st.markdown("#### System Health")
    compliance = client.compliance_status()
    llm = client.llm_health()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("LLM Status", llm.get("status", "Unknown"))
    with c2:
        st.metric("Consent Enforced", "Yes" if compliance.get("consent_enforced") else "No")
    with c3:
        st.metric("Policy Version", compliance.get("safety_policy_version", "v1"))

    st.markdown("#### RAG Index Health")
    index_health = client.index_health()
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.metric("Index Version", index_health.get("index_version", "N/A"))
    with i2:
        st.metric("Documents", int(index_health.get("document_count", 0)))
    with i3:
        st.metric("Chunks", int(index_health.get("chunk_count", 0)))
    with i4:
        st.metric("Embedding Mode", index_health.get("embedding_mode", "unknown"))
    st.caption(
        "Cache hit rates | "
        f"embedding: {index_health.get('cache_hit_rates', {}).get('embedding', 0.0):.2%}, "
        f"answer: {index_health.get('cache_hit_rates', {}).get('answer', 0.0):.2%}, "
        f"router: {index_health.get('cache_hit_rates', {}).get('router', 0.0):.2%}"
    )
    if st.button("Force Reindex", key="admin_force_reindex"):
        result = client.rebuild_index(refresh_prompts=False)
        health = result.get("index_health", {})
        st.success(
            "Reindex completed. "
            f"Version {health.get('index_version', 'N/A')} | "
            f"Chunks {health.get('chunk_count', 0)}"
        )
        st.rerun()

    st.markdown("#### SLA Metrics")
    sla = client.metrics_sla()
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("P50 Latency", f"{sla.get('p50_latency_ms', 0)} ms")
    with k2:
        st.metric("P95 Latency", f"{sla.get('p95_latency_ms', 0)} ms")
    with k3:
        st.metric("Error Rate", f"{sla.get('error_rate', 0):.2%}")
    with k4:
        st.metric("Fallback Rate", f"{sla.get('fallback_rate', 0):.2%}")
    with k5:
        st.metric("Session Completion", f"{sla.get('session_completion_rate', 0):.2%}")

    st.markdown("#### Monthly Snapshots & Session Rollups")
    snapshots_payload = client.monthly_snapshots()
    snapshots = snapshots_payload.get("snapshots", [])
    rollups = snapshots_payload.get("session_rollups", [])
    if snapshots:
        st.dataframe(pd.DataFrame(snapshots), use_container_width=True)
    else:
        st.info("No monthly snapshots yet.")
    if rollups:
        st.dataframe(pd.DataFrame(rollups), use_container_width=True)
    else:
        st.info("No session rollups yet.")

    st.markdown("#### Trace Explorer")
    trace_input = st.text_input("Trace ID", key="trace_input_admin")
    if st.button("Load Trace", key="load_trace_admin") and trace_input:
        st.session_state.trace_lookup_result = client.get_trace(trace_input)

    if st.session_state.trace_lookup_result:
        st.json(st.session_state.trace_lookup_result)

    st.markdown("#### Governance Lifecycle")
    gov_payload = client.governance_items(limit=200)
    gov_items = gov_payload.get("items", [])
    gov_counts = gov_payload.get("counts", {})
    st.write(f"Counts: {gov_counts}")

    if gov_items:
        gov_df = pd.DataFrame(gov_items)
        st.dataframe(gov_df, use_container_width=True)

        selected_id = st.number_input("Item ID to update", min_value=0, step=1, key="gov_item_id")
        new_status = st.selectbox("New Status", options=["Draft", "Approved", "Deprecated"], key="gov_status")
        if st.button("Update Governance Status", key="gov_update"):
            if selected_id > 0:
                client.update_governance_status(int(selected_id), new_status)
                st.success("Governance status updated.")
                st.rerun()
            else:
                st.warning("Enter a valid item ID.")
    else:
        st.info("No governance items found.")


st.set_page_config(page_title="PMG Intelligence Chatbot", layout="wide")
apply_pmg_theme()
_init_state()

config = load_config()
data_dir = config.excel_source_path.parent if config.excel_source_path.parent.exists() else (Path.cwd() / "data")
uploads_dir = data_dir / "user_uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)

try:
    with st.spinner("Loading runtime..."):
        artifacts = build_runtime(
            force_rebuild_cache=st.session_state.force_rebuild_cache,
            force_refresh_prompts=st.session_state.force_refresh_prompts,
        )
        st.session_state.force_rebuild_cache = False
        st.session_state.force_refresh_prompts = False
except Exception as exc:  # noqa: BLE001
    st.error(f"Startup failed: {exc}")
    st.stop()

client = ApiClient(config=config, artifacts=artifacts)
_ensure_chat_session_enabled(client)

if st.session_state.pending_toast:
    st.toast(st.session_state.pending_toast)
    st.session_state.pending_toast = None

cache_startup, cache_current, cache_action, cache_built = _cache_status_text(artifacts)
llm_health = client.llm_health()
llm_status = llm_health.get("status", "Unknown")
deployment_mode = "Remote API" if client.using_remote_api else "In-process API adapter"
llm_badge = "🟢 Connected" if str(llm_status).lower() == "connected" else "🔴 Disconnected"

header_logo_col, header_menu_col, header_title_col, header_dashboard_col = st.columns([1.1, 1.2, 4.5, 1.4])
with header_logo_col:
    logo_path = Path(__file__).parent / "assets" / "pmg_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=130)
    else:
        st.markdown("### PMG")

with header_menu_col:
    with st.popover("Chat & System"):
        export_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        st.markdown("#### Chat Export")
        st.download_button(
            "Export Chat (.txt)",
            data=_chat_history_text(st.session_state.chat_history),
            file_name=f"chat_export_{export_ts}.txt",
            mime="text/plain",
            use_container_width=True,
            key="export_chat_txt",
        )
        st.download_button(
            "Export Chat (.json)",
            data=json.dumps(st.session_state.chat_history, indent=2),
            file_name=f"chat_export_{export_ts}.json",
            mime="application/json",
            use_container_width=True,
            key="export_chat_json",
        )
        st.markdown("#### LLM Health")
        st.write(llm_badge)
        if llm_health.get("last_check_time"):
            st.caption(f"Checked: {llm_health.get('last_check_time')}")

        st.markdown("#### Client Mode")
        st.write(f"**{deployment_mode}**")
        if client.using_remote_api:
            st.caption("Remote API means this UI calls your FastAPI backend over HTTP using API_BASE_URL.")
        else:
            st.caption("In-process adapter means this UI runs backend logic locally without HTTP calls.")

        st.markdown("#### Cache Controls")
        st.caption(f"Startup: {cache_startup} | Current: {cache_current} | Action: {cache_action}")
        st.caption(f"Last Build: {cache_built}")
        if st.button("Force Rebuild Cache", use_container_width=True, key="menu_force_rebuild"):
            _request_cache_action(
                rebuild=True,
                refresh_prompts=False,
                toast_message="Cache rebuild requested.",
            )
        if st.button("Refresh Starter Prompts", use_container_width=True, key="menu_refresh_prompts"):
            _request_cache_action(
                rebuild=False,
                refresh_prompts=True,
                toast_message="Starter prompts refresh requested.",
            )

with header_title_col:
    st.title("Research Data Dictionary Chatbot")
    st.caption("Single source of truth for variables, labels, mappings, and trends")

with header_dashboard_col:
    with st.container(border=True):
        st.markdown("###### Dashboard")
        st.toggle("Admin & Monitoring", key="show_admin_dashboard")
        st.caption("Admin View" if st.session_state.show_admin_dashboard else "Chat View")

st.divider()

survey_filter = "All"
wave_filter = "All"
topic_filter = "All"
label_type_filter = "All"

with st.sidebar:
    st.markdown("### Controls")

    with st.expander("Add Files", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload additional .xlsx or .docx files",
            type=["xlsx", "docx"],
            accept_multiple_files=True,
            key="sidebar_uploader",
        )
        if st.button("Add Uploaded Files", use_container_width=True, key="sidebar_add_files"):
            saved_count = _save_uploaded_files(uploads_dir, uploaded_files)
            if saved_count > 0:
                _request_cache_action(
                    rebuild=True,
                    refresh_prompts=True,
                    toast_message=f"Added {saved_count} file(s). Cache and prompts refreshed.",
                )
            else:
                st.toast("No files selected.")

    with st.expander("Utilities", expanded=False):
        st.session_state.llm_provider = st.selectbox(
            "LLM Provider",
            options=["chatgpt"],
            format_func=lambda x: "ChatGPT (OpenAI)",
            index=0,
        )

        chatgpt_models = list(config.openai_chat_models)
        if not chatgpt_models:
            chatgpt_models = [config.openai_chat_model]
        if st.session_state.llm_model not in chatgpt_models:
            st.session_state.llm_model = chatgpt_models[0]
        st.session_state.llm_model = st.selectbox(
            "ChatGPT Model",
            options=chatgpt_models,
            index=chatgpt_models.index(st.session_state.llm_model),
        )

    with st.expander("Inference Parameters", expanded=False):
        st.session_state.inference_max_length_enabled = st.toggle(
            "Max Length",
            value=st.session_state.inference_max_length_enabled,
        )
        st.session_state.inference_max_length = int(
            st.number_input(
                "Max Length (tokens)",
                min_value=64,
                max_value=8192,
                step=64,
                value=int(st.session_state.inference_max_length),
                disabled=not st.session_state.inference_max_length_enabled,
            )
        )

        st.session_state.inference_top_p_enabled = st.toggle(
            "Top p",
            value=st.session_state.inference_top_p_enabled,
        )
        st.session_state.inference_top_p = st.slider(
            "Top p value",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(st.session_state.inference_top_p),
            disabled=not st.session_state.inference_top_p_enabled,
        )

        st.session_state.inference_temperature_enabled = st.toggle(
            "Temperature",
            value=st.session_state.inference_temperature_enabled,
        )
        st.session_state.inference_temperature = st.slider(
            "Temperature value",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            value=float(st.session_state.inference_temperature),
            disabled=not st.session_state.inference_temperature_enabled,
        )

    with st.expander("Input Method", expanded=False):
        input_methods = ["Document", "Webpage", "Audio", "Image", "PPT"]
        if st.session_state.input_method not in input_methods:
            st.session_state.input_method = "Document"
        st.session_state.input_method = st.radio(
            "Choose Input Method",
            options=input_methods,
            index=input_methods.index(st.session_state.input_method),
        )

    with st.expander("Translation Method", expanded=False):
        st.session_state.translation_method = st.selectbox(
            "Method",
            options=["Built-in Dictionary"],
            index=0,
        )
        st.session_state.language = st.selectbox(
            "Language",
            options=["en", "fr"],
            format_func=lambda x: "English" if x == "en" else "Francais",
            index=0 if st.session_state.language == "en" else 1,
        )

    with st.expander("Route Type", expanded=True):
        st.session_state.router_mode = st.selectbox(
            "Route",
            options=["hybrid", "llm", "deterministic"],
            index=["hybrid", "llm", "deterministic"].index(st.session_state.router_mode),
        )
        st.caption("Dynamic Filter Toggles")
        survey_filter = st.selectbox("Survey", options=["All"] + artifacts.surveys, index=0)
        wave_filter = st.selectbox("Wave/Year", options=["All"] + artifacts.waves, index=0)
        topic_filter = st.selectbox("Topic Label", options=["All"] + artifacts.topics, index=0)
        label_type_filter = st.selectbox("Label Type", options=["All"] + artifacts.topic_source_types, index=0)

    with st.expander("Cache Status & Actions", expanded=False):
        st.caption(f"Startup: {cache_startup}")
        st.caption(f"Current: {cache_current}")
        st.caption(f"Action: {cache_action}")
        st.caption(f"Last Build: {cache_built}")
        if st.button("Force Rebuild Cache", use_container_width=True, key="sidebar_force_rebuild"):
            _request_cache_action(
                rebuild=True,
                refresh_prompts=False,
                toast_message="Cache rebuild requested.",
            )
        if st.button("Refresh Starter Prompts", use_container_width=True, key="sidebar_refresh_prompts"):
            _request_cache_action(
                rebuild=False,
                refresh_prompts=True,
                toast_message="Starter prompts refresh requested.",
            )

if st.session_state.show_admin_dashboard:
    _render_admin_dashboard(client)
else:
    library = client.question_library()
    top_questions = library.get("top_questions", [])
    starter_prompt_pool = _combine_starter_prompts(artifacts.starter_prompts, top_questions)

    with st.expander("Starter Prompts", expanded=False):
        starter_search = st.text_input("Search starter prompts", key="starter_prompt_search")
        search_term = starter_search.strip().lower()
        filtered_prompts = (
            [prompt for prompt in starter_prompt_pool if search_term in prompt.lower()]
            if search_term
            else starter_prompt_pool
        )
        st.caption(f"{len(filtered_prompts)} prompt(s)")
        prompt_col_a, prompt_col_b = st.columns(2)
        for idx, prompt in enumerate(filtered_prompts[:60], start=1):
            preview = prompt if len(prompt) <= 84 else f"{prompt[:81]}..."
            target_col = prompt_col_a if (idx - 1) % 2 == 0 else prompt_col_b
            with target_col:
                if st.button(f"{idx}. {preview}", key=f"starter_{idx}", use_container_width=True):
                    st.session_state.prefill_prompt = prompt
        if not filtered_prompts:
            st.info("No prompts match your search.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    prompt = st.chat_input("Ask about variables, labels, mappings, and trends")
    if not prompt and st.session_state.prefill_prompt:
        prompt = st.session_state.prefill_prompt
        st.session_state.prefill_prompt = ""

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        filters = {
            "survey_name": None if survey_filter == "All" else survey_filter,
            "wave_year": None if wave_filter == "All" else wave_filter,
            "topic_label": None if topic_filter == "All" else topic_filter,
            "topic_source_type": None if label_type_filter == "All" else label_type_filter,
        }
        inference_settings: dict[str, int | float] = {}
        if st.session_state.inference_max_length_enabled:
            inference_settings["max_length"] = int(st.session_state.inference_max_length)
        if st.session_state.inference_top_p_enabled:
            inference_settings["top_p"] = float(st.session_state.inference_top_p)
        if st.session_state.inference_temperature_enabled:
            inference_settings["temperature"] = float(st.session_state.inference_temperature)
        try:
            result = client.agent_router(
                session_id=st.session_state.session_id,
                query=prompt,
                language=st.session_state.language,
                filters=filters,
                mode=st.session_state.router_mode,
                llm_provider=st.session_state.llm_provider,
                llm_model=st.session_state.llm_model,
                inference=inference_settings,
                input_method=st.session_state.input_method.lower(),
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent router failed: {exc}")
        else:
            trace_id = result.get("trace_id", "")
            route_used = result.get("route_used", "unknown")
            fallback_used = result.get("fallback_used", False)
            latency_ms = result.get("latency_ms", 0.0)
            answer = result.get("response", "")

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(f"route={route_used} | fallback={fallback_used} | latency={latency_ms}ms | trace_id={trace_id}")
                cards = result.get("cards", [])
                if cards:
                    st.markdown("#### Retrieved Question Cards")
                    for idx, record in enumerate(cards, start=1):
                        with st.expander(f"{idx}. {record['question_id']} | {record['question_text'][:90]}"):
                            st.write(f"**Question ID:** {record['question_id']}")
                            st.write(f"**Question Text:** {record['question_text']}")
                            st.write(f"**Survey:** {record['survey_name']}")
                            st.write(f"**Wave/Year:** {record['wave_year']}")
                            st.write(f"**Topic Labels:** {', '.join(record['topic_labels'])}")
                            st.write(f"**Label Sources:** {record['topic_label_sources']}")
                            st.write(f"**Measurement Level:** {record['measurement_level']}")
                            st.write(f"**Source File:** {record['source_file']}")

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "meta": f"route={route_used} | fallback={fallback_used} | latency={latency_ms}ms | trace_id={trace_id}",
                }
            )
