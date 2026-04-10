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
from ui.chat_utils import build_conversation_context, chat_history_markdown, citation_markers, confidence_badge
from ui.style import apply_pmg_theme


@lru_cache(maxsize=1)
def build_runtime(force_rebuild_cache: bool = False, force_refresh_prompts: bool = False) -> BootstrapArtifacts:
    """Builds UI runtime artifacts (cached) via backend bootstrap."""
    config = load_config()
    return BootstrapService(config).build(
        force_rebuild_cache=force_rebuild_cache,
        force_refresh_prompts=force_refresh_prompts,
    )


def _init_state() -> None:
    """Initializes all Streamlit session state defaults."""
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
    if "user_role" not in st.session_state:
        st.session_state.user_role = "viewer"
    if "survey_filter" not in st.session_state:
        st.session_state.survey_filter = "All"
    if "wave_filter" not in st.session_state:
        st.session_state.wave_filter = "All"
    if "topic_filter" not in st.session_state:
        st.session_state.topic_filter = "All"
    if "label_type_filter" not in st.session_state:
        st.session_state.label_type_filter = "All"


def _save_uploaded_files(uploads_dir: Path, uploaded_files: list[object] | None) -> int:
    """Persists uploaded files and returns count saved."""
    saved_count = 0
    for file in uploaded_files or []:
        target = uploads_dir / file.name
        target.write_bytes(file.getbuffer())
        saved_count += 1
    return saved_count


def _cache_status_text(artifacts: BootstrapArtifacts) -> tuple[str, str, str, str]:
    """Formats cache status summary values for UI display."""
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
    """Schedules cache/prompt refresh action and triggers Streamlit rerun."""
    st.session_state.force_rebuild_cache = rebuild
    st.session_state.force_refresh_prompts = refresh_prompts
    st.session_state.pending_toast = toast_message
    build_runtime.cache_clear()
    st.rerun()


def _ensure_chat_session_enabled(client: ApiClient) -> None:
    """Best-effort auto-consent call for chat session activation.

    Guards against the Streamlit rerender loop: the consent API is called at
    most once per session_id.  Subsequent rerenders (widget interactions,
    cache actions) skip the call entirely via the session_consent_applied flag.
    """
    if st.session_state.get("session_consent_applied"):
        return
    if st.session_state.get("consent_granted"):
        # Already granted in a prior render — mark applied and skip the API call.
        st.session_state.session_consent_applied = True
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
        st.session_state.pending_toast = "Could not auto-enable chat session consent."


def _combine_starter_prompts(starter_prompts: list[str], top_questions: list[str]) -> list[str]:
    """Merges and deduplicates starter prompts for UX."""
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


def _normalize_sidebar_options(values: list[str]) -> list[str]:
    """Normalizes sidebar options into unique, sorted, non-empty strings."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return sorted(out)


def _chat_history_text(history: list[dict[str, object]]) -> str:
    """Serializes chat history into plain-text transcript format."""
    lines: list[str] = []
    for msg in history:
        role = str(msg.get("role", "unknown")).upper()
        if role == "ASSISTANT" and msg.get("question") and msg.get("answer"):
            content = f"Question: {str(msg.get('question')).strip()}\nAnswer: {str(msg.get('answer')).strip()}"
        else:
            content = str(msg.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines) if lines else "No chat messages yet."


def _format_question_answer_block(question: str, answer: str) -> str:
    """Formats assistant response with explicit Question then Answer sections."""
    return f"**Question asked**\n\n{question}\n\n**Answer**\n\n{answer}"


def _should_show_sample_cards(question: str, cards: list[dict[str, object]]) -> bool:
    """Shows sample cards only when user intent implies examples/samples."""
    if not cards:
        return False
    normalized = question.lower()
    trigger_terms = (
        "sample",
        "example",
        "examples",
        "show cards",
        "card",
        "list options",
        "show matches",
    )
    return any(term in normalized for term in trigger_terms)


def _render_sample_cards(cards: list[dict[str, object]]) -> None:
    """Renders compact tile-like cards for sample results."""
    for idx, record in enumerate(cards[:6], start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}. {record['question_id']}**")
            st.write(record["question_text"])
            st.caption(
                f"Survey: {record['survey_name']} | Wave/Year: {record['wave_year']} | "
                f"Measurement: {record['measurement_level']}"
            )
            st.caption(f"Topics: {', '.join(record['topic_labels'])}")


def _render_sharepoint_panel(uploads_dir: Path) -> None:
    """Renders the SharePoint connector form inside the sidebar expander.

    Allows admins to pull ``.xlsx`` and ``.docx`` files directly from a
    SharePoint document library into the local ``user_uploads/`` directory.
    After a successful sync the admin can trigger a cache rebuild so the
    newly downloaded files are ingested immediately.

    Configuration is read from environment variables (``SHAREPOINT_*``).
    Values entered in the form fields override the env vars for the current
    sync only — they are NOT written back to ``.env``.

    Parameters
    ----------
    uploads_dir:
        Path to ``data/user_uploads/`` where downloaded files are saved.
    """
    from backend.loaders.sharepoint_loader import SharePointConfig, SharePointLoader

    st.markdown("##### SharePoint Connector")
    st.caption(
        "Connect to a SharePoint document library to pull survey files directly "
        "into the chatbot. Fill in the fields below or set the corresponding "
        "``SHAREPOINT_*`` environment variables."
    )

    # Pre-populate fields from env vars so the form is ready if already configured.
    default_cfg = SharePointConfig.from_env()

    with st.form("sharepoint_sync_form"):
        sp_tenant = st.text_input(
            "Tenant ID",
            value=default_cfg.tenant_id,
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="Azure AD tenant GUID (SHAREPOINT_TENANT_ID).",
        )
        sp_client_id = st.text_input(
            "Client ID",
            value=default_cfg.client_id,
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="Azure AD application (client) ID (SHAREPOINT_CLIENT_ID).",
        )
        sp_client_secret = st.text_input(
            "Client Secret",
            value=default_cfg.client_secret,
            type="password",
            help="Leave blank to use Device Flow authentication (SHAREPOINT_CLIENT_SECRET).",
        )
        sp_site_url = st.text_input(
            "Site URL",
            value=default_cfg.site_url,
            placeholder="https://contoso.sharepoint.com/sites/ResearchData",
            help="Full SharePoint site URL (SHAREPOINT_SITE_URL).",
        )
        sp_library_path = st.text_input(
            "Library Path",
            value=default_cfg.library_path,
            placeholder="/sites/ResearchData/Shared Documents/Survey Dictionaries",
            help="Server-relative path to the document library folder (SHAREPOINT_LIBRARY_PATH).",
        )
        auth_mode_options = ["client_credentials", "device_flow"]
        sp_auth_mode = st.selectbox(
            "Auth Mode",
            options=auth_mode_options,
            index=auth_mode_options.index(default_cfg.auth_mode)
            if default_cfg.auth_mode in auth_mode_options
            else 0,
            help="client_credentials for service accounts; device_flow opens a browser prompt.",
        )
        rebuild_after = st.checkbox(
            "Rebuild index after sync",
            value=True,
            help="Automatically rebuild the RAG index so new files are searchable immediately.",
        )
        submitted = st.form_submit_button("Sync from SharePoint", use_container_width=True)

    if submitted:
        override_cfg = SharePointConfig(
            tenant_id=sp_tenant.strip(),
            client_id=sp_client_id.strip(),
            client_secret=sp_client_secret.strip(),
            site_url=sp_site_url.strip(),
            library_path=sp_library_path.strip(),
            auth_mode=sp_auth_mode,
        )
        valid, reason = SharePointLoader(config=override_cfg, downloads_dir=uploads_dir).validate_config()
        if not valid:
            st.error(f"SharePoint config incomplete: {reason}")
        else:
            with st.spinner("Connecting to SharePoint…"):
                try:
                    loader = SharePointLoader(config=override_cfg, downloads_dir=uploads_dir)
                    result = loader.sync()
                    if result.errors:
                        for fname, err in result.errors:
                            st.warning(f"Error downloading {fname}: {err}")
                    if result.downloaded:
                        st.success(result.summary)
                        if rebuild_after:
                            _request_cache_action(
                                rebuild=True,
                                refresh_prompts=True,
                                toast_message=f"SharePoint sync complete. {len(result.downloaded)} file(s) added. Rebuilding index…",
                            )
                    else:
                        st.info(result.summary)
                except ImportError as exc:
                    st.error(str(exc))
                except RuntimeError as exc:
                    st.error(f"SharePoint connection failed: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected error during SharePoint sync: {exc}")


def _fallback_reason_text(reason: str | None) -> str:
    """Maps fallback reason codes to concise UI text."""
    if not reason:
        return ""
    mapping = {
        "llm_unavailable": "LLM unavailable",
        "unsupported_provider": "Provider unsupported",
        "safety_fallback": "Safety fallback",
    }
    return mapping.get(reason, reason.replace("_", " ").title())


def _render_unanswered_analytics(client: ApiClient) -> None:
    """Renders unanswered-query analytics table and summary."""
    analytics = client.unanswered_analytics(limit=50)
    counts = analytics.get("counts", {})
    c1, c2 = st.columns(2)
    with c1:
        st.metric("No Cards", int(counts.get("no_cards", 0)))
    with c2:
        st.metric("Clarifier Only", int(counts.get("clarifier_only", 0)))

    recent = analytics.get("recent", [])
    if recent:
        st.markdown("##### Recent Unanswered")
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
    else:
        st.info("No unanswered-query records yet.")

    top_patterns = analytics.get("top_patterns", [])
    if top_patterns:
        st.markdown("##### Top Unanswered Patterns")
        st.dataframe(pd.DataFrame(top_patterns), use_container_width=True)


def _render_admin_dashboard(client: ApiClient, user_role: str) -> None:
    """Renders role-aware monitoring and admin controls."""
    st.markdown("### Admin & Monitoring Dashboard")

    st.markdown("#### System Health")
    llm = client.llm_health()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("LLM Status", llm.get("status", "Unknown"))
    with c2:
        st.metric("Role", user_role.title())

    st.markdown("#### Unanswered Query Analytics")
    _render_unanswered_analytics(client)

    if user_role != "admin":
        st.caption("Analyst view: read-only monitoring features.")
        return

    st.markdown("#### Compliance")
    compliance = client.compliance_status()
    p1, p2 = st.columns(2)
    with p1:
        st.metric("Consent Enforced", "Yes" if compliance.get("consent_enforced") else "No")
    with p2:
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

try:
    me_payload = client.auth_me()
    st.session_state.user_role = str(me_payload.get("role", "viewer"))
except Exception:  # noqa: BLE001
    st.session_state.user_role = "viewer"

if st.session_state.pending_toast:
    st.toast(st.session_state.pending_toast)
    st.session_state.pending_toast = None

user_role = st.session_state.user_role
is_admin = user_role == "admin"
can_dashboard = user_role in {"analyst", "admin"}
can_export = user_role in {"analyst", "admin"}
can_feedback = user_role in {"analyst", "admin"}

cache_startup, cache_current, cache_action, cache_built = _cache_status_text(artifacts)
llm_health = client.llm_health()
llm_status = llm_health.get("status", "Unknown")
deployment_mode = "Remote API" if client.using_remote_api else "In-process API adapter"
llm_badge = "🟢 Connected" if str(llm_status).lower() == "connected" else "🔴 Disconnected"

header_logo_col, header_menu_col, header_title_col, header_dashboard_col = st.columns([1.1, 1.35, 4.35, 1.2])
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
        if can_export:
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
            st.download_button(
                "Export Chat (.md)",
                data=chat_history_markdown(st.session_state.chat_history),
                file_name=f"chat_export_{export_ts}.md",
                mime="text/markdown",
                use_container_width=True,
                key="export_chat_md",
            )
        else:
            st.caption("Export is available for analyst/admin roles.")

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

        st.markdown("#### Role")
        st.write(f"**{user_role.title()}**")

        st.markdown("#### Cache Controls")
        st.caption(f"Startup: {cache_startup} | Current: {cache_current} | Action: {cache_action}")
        st.caption(f"Last Build: {cache_built}")
        if is_admin:
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
        else:
            st.caption("Cache actions are admin-only.")

with header_title_col:
    st.title("Research Data Dictionary Chatbot")
    st.caption("Single source of truth for variables, labels, mappings, and trends")

with header_dashboard_col:
    with st.container(border=True):
        st.markdown("###### Dashboard")
        if can_dashboard:
            st.toggle("Admin & Monitoring", key="show_admin_dashboard")
            st.caption("Admin View" if st.session_state.show_admin_dashboard else "Chat View")
        else:
            st.caption("Viewer mode")

st.divider()

survey_filter = "All"
wave_filter = "All"
topic_filter = "All"
label_type_filter = "All"

with st.sidebar:
    st.markdown("### Controls")
    st.caption(f"Role: {user_role.title()}")

    if is_admin:
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

        # ----------------------------------------------------------------
        # SharePoint connector panel (admin-only)
        # ----------------------------------------------------------------
        with st.expander("SharePoint Connector", expanded=False):
            _render_sharepoint_panel(uploads_dir)

    if user_role in {"analyst", "admin"}:
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
            st.session_state.inference_max_length = max(64, min(8192, int(st.session_state.inference_max_length)))
            st.session_state.inference_top_p = max(0.0, min(1.0, float(st.session_state.inference_top_p)))
            st.session_state.inference_temperature = max(0.0, min(2.0, float(st.session_state.inference_temperature)))
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
            if st.session_state.inference_max_length_enabled:
                st.caption(f"Active max output tokens: {int(st.session_state.inference_max_length)}")
            else:
                st.caption("Active max output tokens: model default")

        with st.expander("Input Method", expanded=False):
            st.session_state.input_method = "Document"
            st.selectbox(
                "Choose Input Method",
                options=["Document"],
                index=0,
                disabled=True,
            )
            st.caption("Webpage/Audio/Image/PPT modes are deprecated until dedicated ingestion pipelines are implemented.")

        with st.expander("Translation Method", expanded=False):
            language_options = ["en", "fr"]
            if st.session_state.language not in language_options:
                st.session_state.language = "en"
            st.session_state.language = st.selectbox(
                "Language",
                options=language_options,
                format_func=lambda x: "English" if x == "en" else "Francais",
                index=0 if st.session_state.language == "en" else 1,
            )
            st.caption("Translation uses the built-in dictionary pipeline.")

    with st.expander("Route Type", expanded=True):
        route_options = ["hybrid", "llm", "deterministic"]
        if user_role in {"analyst", "admin"}:
            if st.session_state.router_mode not in route_options:
                st.session_state.router_mode = "hybrid"
            st.session_state.router_mode = st.selectbox(
                "Route",
                options=route_options,
                index=route_options.index(st.session_state.router_mode),
            )
        else:
            st.session_state.router_mode = "hybrid"
            st.caption("Viewer role uses hybrid route mode.")

        survey_options = ["All", *_normalize_sidebar_options(artifacts.surveys)]
        wave_options = ["All", *_normalize_sidebar_options(artifacts.waves)]
        topic_options = ["All", *_normalize_sidebar_options(artifacts.topics)]
        label_type_options = ["All", *_normalize_sidebar_options(artifacts.topic_source_types)]
        if st.session_state.survey_filter not in survey_options:
            st.session_state.survey_filter = "All"
        if st.session_state.wave_filter not in wave_options:
            st.session_state.wave_filter = "All"
        if st.session_state.topic_filter not in topic_options:
            st.session_state.topic_filter = "All"
        if st.session_state.label_type_filter not in label_type_options:
            st.session_state.label_type_filter = "All"

        st.caption("Dynamic Filter Toggles")
        survey_filter = st.selectbox("Survey", options=survey_options, index=survey_options.index(st.session_state.survey_filter))
        st.session_state.survey_filter = survey_filter
        wave_filter = st.selectbox("Wave/Year", options=wave_options, index=wave_options.index(st.session_state.wave_filter))
        st.session_state.wave_filter = wave_filter
        topic_filter = st.selectbox("Topic Label", options=topic_options, index=topic_options.index(st.session_state.topic_filter))
        st.session_state.topic_filter = topic_filter
        label_type_filter = st.selectbox(
            "Label Type",
            options=label_type_options,
            index=label_type_options.index(st.session_state.label_type_filter),
        )
        st.session_state.label_type_filter = label_type_filter

    with st.expander("Cache Status", expanded=False):
        st.caption(f"Startup: {cache_startup}")
        st.caption(f"Current: {cache_current}")
        st.caption(f"Action: {cache_action}")
        st.caption(f"Last Build: {cache_built}")
        if is_admin:
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

if st.session_state.show_admin_dashboard and can_dashboard:
    _render_admin_dashboard(client, user_role)
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

    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(str(msg.get("role", "assistant"))):
            role = str(msg.get("role", "assistant"))
            if role == "assistant" and msg.get("question") and msg.get("answer"):
                # Low-confidence warning for history messages
                hist_confidence = msg.get("confidence_score")
                if isinstance(hist_confidence, (int, float)) and hist_confidence < 0.35:
                    st.warning(
                        f"Low retrieval confidence ({hist_confidence:.0%}) — results may not be a precise match.",
                        icon="⚠️",
                    )
                st.markdown(_format_question_answer_block(str(msg.get("question")), str(msg.get("answer"))))
                # Follow-up chip in history
                hist_follow_up = msg.get("follow_up_suggestion")
                if hist_follow_up:
                    st.markdown("---")
                    st.caption("💡 Suggested follow-up:")
                    if st.button(hist_follow_up, key=f"hist_followup_{idx}", use_container_width=False):
                        st.session_state.prefill_prompt = hist_follow_up
                        st.rerun()
            else:
                st.markdown(str(msg.get("content", "")))

            meta = str(msg.get("meta", "")).strip()

            if role == "assistant":
                confidence_value = msg.get("confidence_score")
                badge_col, meta_col = st.columns([1, 3])
                with badge_col:
                    st.caption(confidence_badge(confidence_value if isinstance(confidence_value, (int, float)) else None))
                with meta_col:
                    if meta:
                        st.caption(meta)

                fallback_reason = _fallback_reason_text(str(msg.get("fallback_reason"))) if msg.get("fallback_reason") else ""
                if fallback_reason:
                    st.caption(f"Fallback: {fallback_reason}")

                sources = msg.get("sources", []) if isinstance(msg.get("sources", []), list) else []
                if sources:
                    markers = citation_markers(sources)
                    if markers:
                        st.caption(f"Citations: {markers}")
                    with st.expander("Sources", expanded=False):
                        for source in sources:
                            if not isinstance(source, dict):
                                continue
                            st.markdown(
                                f"{source.get('marker', '')} **{source.get('label', '')}**\n\n"
                                f"{source.get('question_text', '')}"
                            )

                if can_feedback and msg.get("trace_id"):
                    trace_id = str(msg.get("trace_id"))
                    score = msg.get("feedback_score")
                    if score in {-1, 1}:
                        label = "Thumbs Up" if score == 1 else "Thumbs Down"
                        st.caption(f"Feedback submitted: {label}")
                    else:
                        note_key = f"feedback_note_{trace_id}_{idx}"
                        note_val = st.text_input("Feedback note (optional)", key=note_key)
                        fb_col_up, fb_col_down = st.columns(2)
                        with fb_col_up:
                            if st.button("Thumbs Up", key=f"fb_up_{trace_id}_{idx}", use_container_width=True):
                                client.submit_feedback(
                                    session_id=st.session_state.session_id,
                                    trace_id=trace_id,
                                    score=1,
                                    note=note_val,
                                )
                                msg["feedback_score"] = 1
                                msg["feedback_note"] = note_val
                                st.toast("Feedback saved.")
                                st.rerun()
                        with fb_col_down:
                            if st.button("Thumbs Down", key=f"fb_down_{trace_id}_{idx}", use_container_width=True):
                                client.submit_feedback(
                                    session_id=st.session_state.session_id,
                                    trace_id=trace_id,
                                    score=-1,
                                    note=note_val,
                                )
                                msg["feedback_score"] = -1
                                msg["feedback_note"] = note_val
                                st.toast("Feedback saved.")
                                st.rerun()

                cards = msg.get("cards", []) if isinstance(msg.get("cards", []), list) else []
                if cards:
                    st.markdown("#### Sample Answer Cards")
                    _render_sample_cards(cards)

    prompt = st.chat_input("Ask about variables, labels, mappings, and trends")
    if not prompt and st.session_state.prefill_prompt:
        prompt = st.session_state.prefill_prompt
        st.session_state.prefill_prompt = ""

    if prompt:
        conversation_context = build_conversation_context(
            st.session_state.chat_history,
            max_turns=config.conversation_memory_turns,
        )
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
                conversation_context=conversation_context,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent router failed: {exc}")
        else:
            trace_id = result.get("trace_id", "")
            route_used = result.get("route_used", "unknown")
            fallback_used = bool(result.get("fallback_used", False))
            latency_ms = result.get("latency_ms", 0.0)
            answer = str(result.get("response", ""))
            cards = result.get("cards", [])
            sources = result.get("sources", [])
            confidence_score = result.get("confidence_score")
            fallback_reason = result.get("fallback_reason")
            answer_mode = str(result.get("answer_mode", "direct_answer"))
            needs_clarification = bool(result.get("needs_clarification", False))
            unanswered_reason = result.get("unanswered_reason")
            follow_up_suggestion = result.get("follow_up_suggestion")

            fallback_reason_text = _fallback_reason_text(str(fallback_reason)) if fallback_reason else ""
            meta = (
                f"route={route_used} | fallback={fallback_used} | latency={latency_ms:.0f}ms | "
                f"trace_id={trace_id} | mode={answer_mode}"
            )
            if needs_clarification:
                meta += " | clarification_required=true"
            if unanswered_reason:
                meta += f" | unanswered={unanswered_reason}"

            show_sample_cards = _should_show_sample_cards(prompt, cards)
            message_cards = cards if show_sample_cards and isinstance(cards, list) else []

            with st.chat_message("assistant"):
                # Low-confidence notice shown above the answer when retrieval was weak
                if isinstance(confidence_score, (int, float)) and confidence_score < 0.35:
                    st.warning(
                        f"Low retrieval confidence ({confidence_score:.0%}) — results may not be a precise match. "
                        "Consider broadening your query or adjusting filters.",
                        icon="⚠️",
                    )

                st.markdown(_format_question_answer_block(prompt, answer))

                # Follow-up suggestion rendered as a clickable prompt chip
                if follow_up_suggestion:
                    st.markdown("---")
                    st.caption("💡 Suggested follow-up:")
                    if st.button(follow_up_suggestion, key=f"followup_{trace_id}", use_container_width=False):
                        st.session_state.prefill_prompt = follow_up_suggestion
                        st.rerun()

                badge_col, meta_col = st.columns([1, 3])
                with badge_col:
                    if isinstance(confidence_score, (int, float)):
                        st.caption(confidence_badge(float(confidence_score)))
                    else:
                        st.caption(confidence_badge(None))
                with meta_col:
                    st.caption(meta)

                if fallback_reason_text:
                    st.caption(f"Fallback: {fallback_reason_text}")

                if isinstance(sources, list) and sources:
                    marker_text = citation_markers(sources)
                    if marker_text:
                        st.caption(f"Citations: {marker_text}")
                    with st.expander("Sources", expanded=False):
                        for source in sources:
                            if not isinstance(source, dict):
                                continue
                            st.markdown(
                                f"{source.get('marker', '')} **{source.get('label', '')}**\n\n"
                                f"{source.get('question_text', '')}"
                            )

                if message_cards:
                    st.markdown("#### Sample Answer Cards")
                    _render_sample_cards(message_cards)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "question": prompt,
                    "answer": answer,
                    "cards": message_cards,
                    "sources": sources if isinstance(sources, list) else [],
                    "trace_id": trace_id,
                    "answer_mode": answer_mode,
                    "needs_clarification": needs_clarification,
                    "unanswered_reason": unanswered_reason,
                    "confidence_score": confidence_score,
                    "fallback_reason": fallback_reason,
                    "follow_up_suggestion": follow_up_suggestion,
                    "meta": meta,
                }
            )
