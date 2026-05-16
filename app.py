from __future__ import annotations

from html import escape
import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# Voice mode removed — voice assistant files deleted on user request


ROOT = Path(__file__).resolve().parent
DATA_PACKAGE_PATH = ROOT / "ghana_data_package.json"
CSV_PATH = ROOT / "Virtue Foundation Ghana v0.3 - Sheet1 (1).csv"
MAP_PATH = ROOT / "ghana_health_map (1).html"
ENV_PATH = ROOT / ".env"
DEFAULT_DASHBOARD_EMBED_URL = "https://dbc-2c6d5247-de8a.cloud.databricks.com/embed/dashboardsv3/01f13f9cd3fb1357b5abd5e152034084?o=7474647497852266"
DEFAULT_DASHBOARD_VIEW_URL = "https://dbc-2c6d5247-de8a.cloud.databricks.com/dashboardsv3/01f13f9cd3fb1357b5abd5e152034084?o=7474647497852266"
DEFAULT_DASHBOARD_CATALOG = [
    {"name": "Overview", "embed_url": DEFAULT_DASHBOARD_EMBED_URL, "view_url": DEFAULT_DASHBOARD_VIEW_URL},
    {"name": "Intervention Planning", "embed_url": DEFAULT_DASHBOARD_EMBED_URL, "view_url": DEFAULT_DASHBOARD_VIEW_URL},
    {"name": "Data Quality", "embed_url": DEFAULT_DASHBOARD_EMBED_URL, "view_url": DEFAULT_DASHBOARD_VIEW_URL},
]


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if " #" in value:
            value = value.split(" #", 1)[0].strip()

        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@st.cache_data(show_spinner=False)
def load_data_package() -> dict:
    if not DATA_PACKAGE_PATH.exists():
        return {}
    return json.loads(DATA_PACKAGE_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_facilities() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()

    frame = pd.read_csv(CSV_PATH, dtype=str, low_memory=False, on_bad_lines="skip")
    frame.columns = [column.strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame


def safe_int(value: object) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def build_data_context(package: dict, facilities: pd.DataFrame) -> str:
    stats = package.get("stats", {})
    regions = package.get("regions", [])

    lines = [
        "GHANA HEALTHCARE INTELLIGENCE DATA CONTEXT",
        "Use only the facts below. If the data does not support a claim, say so clearly.",
        "",
            f"- Total facilities: {safe_int(stats.get('total'))}",
        f"- Hospitals: {safe_int(stats.get('hospitals'))}",
        f"- Clinics: {safe_int(stats.get('clinics'))}",
        f"- NGOs: {safe_int(stats.get('ngos'))}",
        f"- Emergency coverage: {safe_int(stats.get('emergency'))}",
        f"- Surgery coverage: {safe_int(stats.get('surgery'))}",
        f"- ICU coverage: {safe_int(stats.get('icu'))}",
        f"- Maternity coverage: {safe_int(stats.get('maternity'))}",
        f"- Laboratory coverage: {safe_int(stats.get('lab'))}",
        f"- Flagged for review: {safe_int(stats.get('flagged'))}",
        "",
        "Regional snapshot:",
    ]

    for region in regions:
        missing = region.get("missing", [])
        missing_text = ", ".join(missing) if missing else "None"
        lines.append(
            f"- {region.get('name', 'Unknown')}: MDI={region.get('mdi', 'n/a')}, "
            f"alert={region.get('alert', 'n/a')}, total={safe_int(region.get('total'))}, "
            f"hospitals={safe_int(region.get('hospitals'))}, emergency={safe_int(region.get('emergency'))}, "
            f"surgery={safe_int(region.get('surgery'))}, ICU={safe_int(region.get('icu'))}, "
            f"maternity={safe_int(region.get('maternity'))}, lab={safe_int(region.get('lab'))}, "
            f"missing={missing_text}"
        )

    if facilities.empty:
        return "\n".join(lines)

    sample_columns = [
        column
        for column in [
            "name",
            "facilityTypeId",
            "address_city",
            "address_stateOrRegion",
            "specialties",
            "procedure",
            "equipment",
            "capability",
            "description",
            "missionStatement",
        ]
        if column in facilities.columns
    ]

    lines.append("")
    lines.append("Facility search sample:")
    for _, row in facilities.head(12).iterrows():
        snippet = " | ".join(
            [
                f"name={row.get('name', '')}",
                f"type={row.get('facilityTypeId', '')}",
                f"city={row.get('address_city', '')}",
                f"region={row.get('address_stateOrRegion', '')}",
            ]
        )
        lines.append(f"- {snippet}")

    lines.append("")
    lines.append("Raw source columns available for follow-up questions:")
    lines.append(", ".join(sample_columns))
    return "\n".join(lines)


def search_facilities(question: str, facilities: pd.DataFrame, limit: int = 6) -> list[dict]:
    if facilities.empty:
        return []

    terms = [term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 2]
    if not terms:
        terms = [question.lower()]

    scored_rows: list[tuple[int, pd.Series]] = []
    searchable_columns = [
        column
        for column in [
            "name",
            "address_city",
            "address_stateOrRegion",
            "facilityTypeId",
            "specialties",
            "procedure",
            "equipment",
            "capability",
            "description",
            "missionStatement",
        ]
        if column in facilities.columns
    ]

    for _, row in facilities.iterrows():
        haystack = " ".join(str(row.get(column, "")) for column in searchable_columns).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored_rows.append((score, row))
        # Voice Assistant view removed

    results: list[dict] = []
    for _, row in scored_rows[:limit]:
        results.append(
            {
                "name": row.get("name", ""),
                "facility_type": row.get("facilityTypeId", ""),
                "city": row.get("address_city", ""),
                "region": row.get("address_stateOrRegion", ""),
                "specialties": row.get("specialties", ""),
                "capability": row.get("capability", ""),
                "description": row.get("description", "")[:220],
            }
        )

    return results


def build_messages(question: str, history: list[dict], context: str, matches: list[dict]) -> list[dict]:
    system_prompt = (
        "You are Ama, the Ghana Healthcare planning agent and assistant. "
        "Carry a natural conversation, remember the recent turn history, and answer in plain English. "
        "Write a clear, conversational 3-5 sentence answer for a non-technical NGO coordinator. "
        "Ground every claim in the provided context and matching facilities. "
        "When the user asks for planning help, give specific, actionable recommendations. "
        "Return only valid JSON with this structure: "
        '{"answer":"string","findings":[{"point":"string","citation":"string"}],'
        '"recommendations":["string"],"confidence":{"level":"HIGH|MEDIUM|LOW","score":0.0,"reason":"string"}}. '
        "Use the regional snapshot and matching facilities when relevant. "
        "Never mention that you are a demo. "
        "If the question cannot be fully answered from the context, say what is missing. "
        "Findings must always include 3-5 factual points with citations to a specific facility or region and exact field/value used. "
        "Recommendations must always include 2-4 concrete actions and each one must include the action, responsible actor, priority tag, and a short implementation step or measurable success metric. "
        "Confidence should explain whether the data is complete, partial, or sparse and should be based on the evidence available."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}"},
    ]

    if matches:
        messages.append({"role": "user", "content": f"Relevant facilities for this question:\n{json.dumps(matches, indent=2)}"})

    for item in history[-6:]:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": json.dumps(item["answer"])})

    messages.append({"role": "user", "content": question})
    return messages


def ask_assistant(question: str, history: list[dict], context: str, matches: list[dict]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "answer": "Set OPENAI_API_KEY in your environment or .env file to enable the assistant.",
            "findings": [],
            "recommendations": ["Add an OpenAI API key and restart the app."],
            "confidence": {"level": "LOW", "score": 0.0, "reason": "Missing API key"},
        }

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=build_messages(question, history, context, matches),
    )

    content = response.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "answer": content,
            "findings": [],
            "recommendations": [],
            "confidence": {"level": "LOW", "score": 0.0, "reason": "Model did not return valid JSON"},
        }

    result.setdefault("answer", "No answer returned.")
    result.setdefault("findings", [])
    result.setdefault("recommendations", [])
    result.setdefault("confidence", {"level": "LOW", "score": 0.0, "reason": "Incomplete model response"})
    return result


def _format_multiline(text: object) -> str:
    return escape(str(text)).replace("\n", "<br>")


def render_welcome_html() -> str:
    return """
    <div class="vf-wrap">
      <div class="vf-chat">
        <div class="vf-welcome">
          <div class="vf-welcome-icon">🏥</div>
          <div class="vf-welcome-title">Ghana Healthcare Planning Assistant</div>
          <div class="vf-welcome-sub">
            I have complete data on 797 facilities across Ghana's 16 regions.<br>
            Every answer includes findings with data citations,<br>
            actionable recommendations, and a confidence assessment.
          </div>
        </div>
      </div>
    </div>
    """


def render_user_message_html(question: str) -> str:
    return f"""
    <div class="vf-wrap">
      <div class="vf-chat">
        <div class="vf-message vf-message-user">
          <div class="vf-avatar-user">🙂</div>
          <div class="vf-message-bubble vf-message-bubble-user">{_format_multiline(question)}</div>
        </div>
      </div>
    </div>
    """


def render_bot_response_html(result: dict) -> str:
    answer = _format_multiline(result.get("answer", ""))
    findings = result.get("findings", [])
    recommendations = result.get("recommendations", [])
    confidence = result.get("confidence", {})

    conf_level = str(confidence.get("level", "MEDIUM"))
    try:
        conf_score = float(confidence.get("score", 0.5))
    except Exception:
        conf_score = 0.5
    conf_reason = _format_multiline(confidence.get("reason", ""))

    conf_color = {"HIGH": "#4caf50", "MEDIUM": "#ff9800", "LOW": "#f44336"}.get(conf_level, "#ff9800")
    conf_bar_pct = max(0, min(100, int(conf_score * 100)))

    html = [
        "<div class=\"vf-wrap\"><div class=\"vf-chat\"><div class=\"vf-response-wrap\">",
        "<div class=\"vf-response-header\"><div class=\"vf-avatar-bot\">🏥</div>",
        "<div class=\"vf-response-label\">Healthcare Assistant · Ghana Intelligence System</div></div>",
        f"<div class=\"vf-card-answer\"><div class=\"vf-card-header\">🔵 &nbsp; ANSWER</div><div class=\"vf-answer-text\">{answer}</div></div>",
    ]

    if findings:
        findings_items = []
        for index, finding in enumerate(findings, 1):
            point = _format_multiline(finding.get("point", ""))
            citation = _format_multiline(finding.get("citation", ""))
            citation_html = f"<div class=\"vf-citation\">📎 {citation}</div>" if citation else ""
            findings_items.append(
                f"<div class=\"vf-finding-item\"><div class=\"vf-finding-point\"><span class=\"vf-finding-num\">{index}</span>{point}</div>{citation_html}</div>"
            )

        html.append(
            f"<div class=\"vf-card-findings\"><div class=\"vf-card-header\">🔎 &nbsp; FINDINGS ({len(findings)} data points)</div>{''.join(findings_items)}</div>"
        )

    if recommendations:
        recommendation_items = []
        for index, recommendation in enumerate(recommendations, 1):
            recommendation_items.append(
                f"<div class=\"vf-rec-item\"><div class=\"vf-rec-num\">{index}</div><div>{_format_multiline(recommendation)}</div></div>"
            )

        html.append(
            f"<div class=\"vf-card-recommendations\"><div class=\"vf-card-header\">✅ &nbsp; RECOMMENDATIONS ({len(recommendations)} actions)</div>{''.join(recommendation_items)}</div>"
        )

    html.append(
        f"<div class=\"vf-card-confidence\"><div class=\"vf-conf-badge\" style=\"background:{conf_color}22;border:1px solid {conf_color};color:{conf_color};\">{escape(conf_level)}</div><div style=\"flex:1;\"><div class=\"vf-conf-bar-wrap\"><div class=\"vf-conf-bar\" style=\"width:{conf_bar_pct}%;background:{conf_color};\"></div></div></div><div class=\"vf-conf-score\">{conf_bar_pct}%</div><div class=\"vf-conf-reason\">{conf_reason}</div></div>"
    )
    html.append("</div></div></div>")
    return "".join(html)


def render_conversation_html(history: list[dict]) -> str:
    if not history:
        return render_welcome_html()

    blocks: list[str] = []
    for entry in history:
        blocks.append(render_user_message_html(entry.get("question", "")))
        blocks.append(render_bot_response_html(entry.get("answer", {})))

    return "".join(blocks)


def render_thinking_html(question: str) -> str:
        return f"""
        <div class="vf-wrap">
            <div class="vf-chat">
                <div class="vf-message vf-message-user">
                    <div class="vf-avatar-user">🙂</div>
                    <div class="vf-message-bubble vf-message-bubble-user">{_format_multiline(question)}</div>
                </div>
                <div class="vf-response-wrap">
                    <div class="vf-response-header">
                        <div class="vf-avatar-bot">🏥</div>
                        <div class="vf-response-label">Healthcare Assistant · Analyzing data...</div>
                    </div>
                    <div class="vf-thinking">⏳ &nbsp; Searching 797 facilities and 16 regions for your answer...</div>
                </div>
            </div>
        </div>
        """


def render_metric(label: str, value: object, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(package: dict) -> None:
    stats = package.get("stats", {})
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-copy">
                <div class="eyebrow">Virtue Foundation Ghana</div>
                <h1>Healthcare intelligence in one shared workspace</h1>
                <p>
                    Ask the LLM about medical access, identify gaps by region, and inspect the
                    interactive map without leaving the page.
                </p>
            </div>
            <div class="hero-badge">Planning workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    with cols[0]:
        render_metric("Facilities", safe_int(stats.get("total")), "Across the current dataset")
    with cols[1]:
        render_metric("Hospitals", safe_int(stats.get("hospitals")), "Facilities with hospital type")
    with cols[2]:
        render_metric("Clinics", safe_int(stats.get("clinics")), "Facilities with clinic type")
    with cols[3]:
        render_metric("Emergency", safe_int(stats.get("emergency")), "Facilities with emergency coverage")
    with cols[4]:
        render_metric("Flagged", safe_int(stats.get("flagged")), "Facilities needing review")


def render_map() -> None:
    if not MAP_PATH.exists():
        st.warning("Map HTML file is missing. Put the exported map next to app.py.")
        return

    map_html = MAP_PATH.read_text(encoding="utf-8")
    components.html(map_html, height=850, scrolling=True)


def load_dashboard_urls() -> list[dict[str, str]]:
    dashboard_config = ROOT / "databricks_dashboards.json"
    if not dashboard_config.exists():
        return DEFAULT_DASHBOARD_CATALOG

    try:
        content = json.loads(dashboard_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(content, list):
        return []

    dashboards: list[dict[str, str]] = []
    for item in content:
        if not isinstance(item, dict) or not item.get("name"):
            continue

        embed_url = item.get("embed_url") or item.get("url") or DEFAULT_DASHBOARD_EMBED_URL
        view_url = item.get("view_url") or item.get("url") or DEFAULT_DASHBOARD_VIEW_URL
        dashboards.append({"name": item["name"], "embed_url": embed_url, "view_url": view_url})

    if dashboards:
        return dashboards

    return DEFAULT_DASHBOARD_CATALOG


def render_dashboard_access_card(name: str, view_url: str) -> None:
    st.markdown(
        f"""
        <div class="dashboard-access-card">
            <div class="dashboard-access-title">{name}</div>
            <div class="dashboard-access-text">
                Databricks embedding is disabled for this workspace. Open the dashboard in Databricks after signing in.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<a class='dashboard-link-button' href=\"{view_url}\" target=\"_blank\" rel=\"noopener noreferrer\">Open {name} in Databricks</a>",
        unsafe_allow_html=True,
    )


def get_dashboard_catalog() -> list[dict[str, str]]:
    catalog = load_dashboard_urls()
    page_names = ["Overview", "Intervention Planning", "Data Quality"]

    if not catalog:
        catalog = DEFAULT_DASHBOARD_CATALOG

    catalog_by_name = {item["name"]: item for item in catalog}
    pages: list[dict[str, str]] = []
    for index, page_name in enumerate(page_names):
        item = catalog_by_name.get(page_name, catalog[0] if catalog else DEFAULT_DASHBOARD_CATALOG[0])
        pages.append(
            {
                "name": page_name,
                "embed_url": item.get("embed_url", DEFAULT_DASHBOARD_EMBED_URL),
                "view_url": item.get("view_url", DEFAULT_DASHBOARD_VIEW_URL),
            }
        )

    return pages


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 199, 0, 0.10), transparent 28%),
                radial-gradient(circle at right top, rgba(16, 107, 255, 0.14), transparent 26%),
                linear-gradient(180deg, #08121f 0%, #0d1726 100%);
            color: #e5eef8;
        }

        .main .block-container {
            background: transparent;
            color: #e5eef8;
        }

        .stApp, .main, .main .block-container {
            min-height: 100vh;
        }

        .hero-shell {
            display: flex;
            justify-content: space-between;
            gap: 24px;
            align-items: flex-end;
            padding: 1.4rem 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            background: linear-gradient(135deg, #08121f 0%, #11253d 55%, #18395d 100%);
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 20px 55px rgba(3, 13, 25, 0.28);
        }

        .hero-copy h1 {
            margin: 0.15rem 0 0.5rem;
            font-size: clamp(1.8rem, 4vw, 3rem);
            line-height: 1.05;
        }

        .hero-copy p {
            margin: 0;
            max-width: 760px;
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: #f5c542;
            font-weight: 700;
        }

        .hero-badge {
            border-radius: 999px;
            padding: 0.7rem 1rem;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.16);
            color: #eaf2ff;
            font-size: 0.88rem;
            white-space: nowrap;
        }

        .metric-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
            min-height: 120px;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #0b2a4a;
            line-height: 1;
        }

        .metric-label {
            margin-top: 0.55rem;
            font-weight: 700;
            color: #17324d;
        }

        .metric-help {
            margin-top: 0.25rem;
            color: #6b7280;
            font-size: 0.84rem;
        }

        .section-card {
            background: white;
            border-radius: 20px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            color: #0f172a;
        }

        .section-card * {
            color: inherit;
        }

        .section-card p,
        .section-card li,
        .section-card span,
        .section-card label,
        .section-card div,
        .section-card h1,
        .section-card h2,
        .section-card h3,
        .section-card h4,
        .section-card h5,
        .section-card h6 {
            color: #0f172a;
        }

        .section-card .stMarkdown,
        .section-card .stCaption,
        .section-card .stDataFrame,
        .section-card .stTable,
        .section-card .stMetric {
            color: #0f172a;
        }

        .section-card [data-testid="stChatMessage"],
        .section-card [data-testid="stChatMessage"] *,
        .section-card .stChatMessage,
        .section-card .stChatMessage * {
            color: #0f172a !important;
        }

        .dashboard-note {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            color: #334155;
            margin-bottom: 1rem;
        }

        .dashboard-warning {
            background: #fff7ed;
            border: 1px solid #ffd8a8;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            color: #92400e;
            margin-bottom: 1rem;
            font-weight: 700;
        }

        .dashboard-access-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1rem;
        }

        .dashboard-access-title {
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }

        .dashboard-access-text {
            color: #475569;
            margin-bottom: 0.9rem;
        }

        .dashboard-link-button {
            display: inline-block;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            background: #0f4c81;
            color: #ffffff !important;
            text-decoration: none;
            font-weight: 700;
        }

        .vf-wrap {
            margin-top: 0.75rem;
        }

        .vf-chat {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }

        .vf-welcome {
            background: #08111d;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1rem 0.95rem;
            color: #eaf2ff;
        }

        .vf-welcome-icon {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }

        .vf-welcome-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .vf-welcome-sub {
            font-size: 0.92rem;
            color: rgba(234,242,255,0.82);
            line-height: 1.65;
        }

        .vf-message {
            display: flex;
            gap: 0.7rem;
            align-items: flex-start;
        }

        .vf-message-bubble {
            flex: 1;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            line-height: 1.65;
            font-size: 0.96rem;
        }

        .vf-message-user {
            justify-content: flex-start;
        }

        .vf-message-bubble-user {
            background: #1f2430;
            border: 1px solid rgba(255,255,255,0.08);
            color: #f3f7ff;
        }

        .vf-avatar-user,
        .vf-avatar-bot {
            width: 2rem;
            height: 2rem;
            min-width: 2rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            line-height: 1;
        }

        .vf-avatar-user {
            background: #ff3b30;
            color: white;
        }

        .vf-avatar-bot {
            background: #ff9800;
            color: white;
        }

        .vf-response-wrap {
            background: #08111d;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem;
            color: #eaf2ff;
        }

        .vf-response-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.85rem;
        }

        .vf-response-label {
            font-size: 0.82rem;
            color: #90a4b8;
            font-weight: 700;
        }

        .vf-card-answer,
        .vf-card-findings,
        .vf-card-recommendations,
        .vf-card-confidence {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
        }

        .vf-thinking {
            color: #c8d7e6;
            padding: 0.4rem 0.1rem 0.15rem;
            font-size: 0.95rem;
        }

        .vf-card-answer {
            background: #0d2137;
            border-left: 3px solid #42a5f5;
        }

        .vf-card-findings {
            background: #040c18;
            border-left: 3px solid #1565c0;
        }

        .vf-card-recommendations {
            background: #040c18;
            border-left: 3px solid #00c853;
        }

        .vf-card-confidence {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: #040c18;
        }

        .vf-card-header {
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 0.65rem;
        }

        .vf-answer-text {
            color: #e3f2fd;
            font-size: 0.96rem;
            line-height: 1.75;
        }

        .vf-finding-item {
            margin-bottom: 0.8rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid #0d2137;
        }

        .vf-finding-item:last-child,
        .vf-rec-item:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }

        .vf-finding-point,
        .vf-citation,
        .vf-rec-item,
        .vf-conf-reason {
            color: #b0bec5;
            font-size: 0.86rem;
            line-height: 1.65;
        }

        .vf-finding-num,
        .vf-rec-num {
            display: inline-flex;
            width: 1.3rem;
            height: 1.3rem;
            margin-right: 0.45rem;
            border-radius: 999px;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.08);
            color: #eaf2ff;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .vf-rec-item {
            display: flex;
            gap: 0.7rem;
            margin-bottom: 0.75rem;
            align-items: flex-start;
        }

        .vf-conf-badge {
            border-radius: 0.5rem;
            padding: 0.28rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 800;
            min-width: 4rem;
            text-align: center;
        }

        .vf-conf-bar-wrap {
            width: 100%;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            overflow: hidden;
            height: 0.5rem;
        }

        .vf-conf-bar {
            height: 100%;
            border-radius: 999px;
        }

        .vf-conf-score {
            font-weight: 800;
            color: #eaf2ff;
            min-width: 3rem;
            text-align: right;
        }

        .stChatMessage {
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Ghana Healthcare Intelligence", page_icon="G", layout="wide")
    load_local_env(ENV_PATH)

    package = load_data_package()
    facilities = load_facilities()
    context = build_data_context(package, facilities)

    inject_styles()
    render_header(package)

    view = st.radio("Choose a view", ["Assistant", "Map", "Dashboards"], horizontal=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    if view == "Assistant":
        left, right = st.columns([1.3, 0.9], gap="large")

        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Ask the assistant")
            st.caption("Use the quick prompts or ask your own question about regions, facilities, or support priorities.")

            quick_prompts = [
                "Which region looks most underserved?",
                "What facilities in Ashanti need support?",
                "Where are emergency capabilities missing?",
                "Summarize the top priorities for a donor briefing.",
            ]
            prompt_cols = st.columns(2)
            for index, prompt in enumerate(quick_prompts):
                if prompt_cols[index % 2].button(prompt, use_container_width=True):
                    st.session_state.pending_question = prompt

            st.markdown(render_conversation_html(st.session_state.history), unsafe_allow_html=True)

            if "pending_question" in st.session_state:
                st.session_state.assistant_input = st.session_state.pop("pending_question")

            question = st.chat_input("Ask about the Ghana healthcare system", key="assistant_input")
            if question:
                matches = search_facilities(question, facilities)
                st.markdown(render_thinking_html(question), unsafe_allow_html=True)
                with st.spinner("Thinking..."):
                    answer = ask_assistant(question, st.session_state.history, context, matches)
                st.session_state.history.append({"question": question, "answer": answer})
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("How to use")
            st.caption("Pick a view, ask a question from the bottom bar, and the answer will stack above with findings, recommendations, and confidence.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Voice Assistant view removed

    elif view == "Map":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Interactive map")
        st.caption("This is the existing Folium map exported from the notebook and embedded directly in the UI.")
        st.caption("Tip: use the legend and layer controls at the top of the map to toggle facility types and overlays.")
        render_map()
        if MAP_PATH.exists():
            st.download_button(
                "Download map HTML",
                data=MAP_PATH.read_bytes(),
                file_name="ghana_health_map.html",
                mime="text/html",
            )
        st.markdown('</div>', unsafe_allow_html=True)

    elif view == "Dashboards":
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Databricks dashboards")
        st.caption("Databricks embedding is disabled in this workspace, so the dashboards open directly in Databricks. Users with access can sign in and view them there.")

        st.markdown(
            "<div class='dashboard-warning'>"
            "This workspace does not allow dashboard embedding. The direct Databricks link below is the reliable way to view each dashboard."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='dashboard-note'>"
            "The app keeps the dashboard links in databricks_dashboards.json so you can replace them without changing the UI code. "
            "If your workspace admin later enables embedding, this view can be changed back to inline iframes."
            "</div>",
            unsafe_allow_html=True,
        )

        dashboards = get_dashboard_catalog()
        page_tabs = st.tabs([item["name"] for item in dashboards])
        for tab, dashboard in zip(page_tabs, dashboards):
            with tab:
                render_dashboard_access_card(dashboard.get("name", "Dashboard"), dashboard.get("view_url", ""))

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()