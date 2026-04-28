from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DATA_PACKAGE_PATH = ROOT / "ghana_data_package.json"
CSV_PATH = ROOT / "Virtue Foundation Ghana v0.3 - Sheet1 (1).csv"
MAP_PATH = ROOT / "ghana_health_map (1).html"
ENV_PATH = ROOT / ".env"


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
        "Summary statistics:",
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

    scored_rows.sort(key=lambda item: (-item[0], str(item[1].get("name", ""))))

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
        "You are the Ghana Healthcare Intelligence assistant for a hackathon demo. "
        "Answer in plain English and keep the answer grounded in the provided context. "
        "Return only valid JSON with this structure: "
        '{"answer":"string","findings":[{"point":"string","citation":"string"}],'
        '"recommendations":["string"],"confidence":{"level":"HIGH|MEDIUM|LOW","score":0.0,"reason":"string"}}. '
        "Use the regional snapshot and matching facilities when relevant. "
        "If the question cannot be fully answered from the context, say what is missing."
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
                <h1>Healthcare intelligence in one shared demo</h1>
                <p>
                    Ask the LLM about medical access, identify gaps by region, and inspect the
                    interactive map without leaving the page.
                </p>
            </div>
            <div class="hero-badge">Hackathon-ready UI</div>
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
        return [
            {
                "name": "Overview",
                "url": "https://dbc-2c6d5247-de8a.cloud.databricks.com/dashboardsv3/01f13f9cd3fb1357b5abd5e152034084?o=7474647497852266&embedded=true",
            }
        ]

    try:
        content = json.loads(dashboard_config.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(content, list):
        return []

    dashboards = [item for item in content if isinstance(item, dict) and item.get("name") and item.get("url")]
    if dashboards:
        return dashboards

    return [
        {
            "name": "Overview",
            "url": "https://dbc-2c6d5247-de8a.cloud.databricks.com/dashboardsv3/01f13f9cd3fb1357b5abd5e152034084?o=7474647497852266&embedded=true",
        }
    ]


def ensure_embedded_dashboard_url(url: str) -> str:
    parsed = urlparse(url)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items["embedded"] = "true"
    normalized = parsed._replace(query=urlencode(query_items, doseq=True))
    return urlunparse(normalized)


def render_dashboard_iframe(url: str) -> None:
    embed_url = ensure_embedded_dashboard_url(url)
    components.html(
        f"""
        <iframe
            src="{embed_url}"
            style="width: 100%; min-height: 760px; border: 0; border-radius: 16px; background: white;"
            loading="lazy"
            referrerpolicy="no-referrer"
            allowfullscreen
        ></iframe>
        """,
        height=800,
        scrolling=True,
    )


def get_dashboard_catalog() -> list[dict[str, str]]:
    catalog = load_dashboard_urls()
    default_url = "https://dbc-2c6d5247-de8a.cloud.databricks.com/dashboardsv3/01f13f9cd3fb1357b5abd5e152034084?o=7474647497852266&embedded=true"
    page_names = ["Overview", "Intervention Planning", "Data Quality"]

    if not catalog:
        catalog = [{"name": page_names[0], "url": default_url}]

    catalog_by_name = {item["name"]: item["url"] for item in catalog}
    pages: list[dict[str, str]] = []
    for index, page_name in enumerate(page_names):
        pages.append(
            {
                "name": page_name,
                "url": catalog_by_name.get(page_name, catalog[0]["url"] if catalog else default_url),
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

        .dashboard-frame {
            width: 100%;
            min-height: 760px;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            overflow: hidden;
            background: #ffffff;
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

            for entry in st.session_state.history:
                with st.chat_message("user"):
                    st.markdown(entry["question"])
                with st.chat_message("assistant"):
                    st.markdown(entry["answer"].get("answer", "No answer returned."))
                    if entry["answer"].get("findings"):
                        with st.expander("Findings"):
                            for finding in entry["answer"].get("findings", []):
                                st.markdown(f"- {finding.get('point', '')}")
                                citation = finding.get("citation", "")
                                if citation:
                                    st.caption(citation)
                    if entry["answer"].get("recommendations"):
                        with st.expander("Recommendations"):
                            for recommendation in entry["answer"].get("recommendations", []):
                                st.markdown(f"- {recommendation}")
                    confidence = entry["answer"].get("confidence", {})
                    st.caption(
                        f"Confidence: {confidence.get('level', 'LOW')} | {confidence.get('score', 0)} | {confidence.get('reason', '')}"
                    )
            

            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Ready-made talking points")
            if package.get("regions"):
                for region in package.get("regions", [])[:6]:
                    with st.container():
                        st.markdown(f"**{region.get('name', 'Unknown')}**")
                        st.caption(
                            f"MDI {region.get('mdi', 'n/a')} | alert {region.get('alert', 'n/a')} | missing: {', '.join(region.get('missing', [])) or 'None'}"
                        )
            else:
                st.info("Regional summaries will appear here after the data package loads.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Move the chat input to the bottom of the Assistant view so it appears after the content
        if "pending_question" in st.session_state:
            st.session_state.assistant_input = st.session_state.pop("pending_question")

        question = st.chat_input("Ask about the Ghana healthcare system", key="assistant_input")
        if question:
            matches = search_facilities(question, facilities)
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_assistant(question, st.session_state.history, context, matches)
                st.markdown(answer.get("answer", "No answer returned."))

                findings = answer.get("findings", [])
                if findings:
                    with st.expander("Findings"):
                        for finding in findings:
                            st.markdown(f"- {finding.get('point', '')}")
                            citation = finding.get("citation", "")
                            if citation:
                                st.caption(citation)

                recommendations = answer.get("recommendations", [])
                if recommendations:
                    with st.expander("Recommendations"):
                        for recommendation in recommendations:
                            st.markdown(f"- {recommendation}")

                confidence = answer.get("confidence", {})
                st.caption(
                    f"Confidence: {confidence.get('level', 'LOW')} | {confidence.get('score', 0)} | {confidence.get('reason', '')}"
                )

            st.session_state.history.append({"question": question, "answer": answer})
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
        st.caption("Three page tabs are shown below. Replace the URLs in databricks_dashboards.json with the exact Databricks page links if you want them to differ.")

        st.markdown(
            "<div class='dashboard-warning'>"
            "Databricks account required: You must be signed in to Databricks to view some dashboards inline. "
            "If a dashboard is private or requires an active Databricks session, the app will show a link you can open in a new tab."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='dashboard-note'>"
            "Only dashboards that are shared and reachable in a browser can be embedded here. "
            "If embedding fails, click the link below to open the page directly in Databricks (new tab). "
            "Ask your Databricks admin to allowlist this app's origin for embed permission if you need inline viewing."
            "</div>",
            unsafe_allow_html=True,
        )

        dashboards = get_dashboard_catalog()
        page_tabs = st.tabs([item["name"] for item in dashboards])
        for tab, dashboard in zip(page_tabs, dashboards):
            with tab:
                url = ensure_embedded_dashboard_url(dashboard.get("url", ""))
                # Show an explicit clickable link that opens in a new tab
                st.markdown(
                    f"<div style='margin-bottom:0.5rem;'>"
                    f"<a href=\"{dashboard.get('url', '')}\" target=\"_blank\" rel=\"noopener noreferrer\" "
                    f"style=\"display:inline-block;padding:0.5rem 0.9rem;border-radius:8px;border:1px solid rgba(255,255,255,0.06);background:rgba(255,255,255,0.02);color:#eaf2ff;text-decoration:none;\">"
                    f"Open {dashboard.get('name')} in Databricks</a>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Also show the embed attempt (iframe). If it fails due to auth, user can use the link above.
                render_dashboard_iframe(url)

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()