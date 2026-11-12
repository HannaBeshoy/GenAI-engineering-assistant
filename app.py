"""Engineering Document Assistant - Streamlit front end.

    streamlit run app.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from assistant import datasheet, quality
from assistant.errors import AssistantError, DemoModeError
from assistant.llm import PROVIDERS, LLMClient, Params, load_demo_answers
from assistant.pdf import read_pdf

load_dotenv()
SAMPLES = Path(__file__).parent / "samples"
SAMPLE_PDF = SAMPLES / "DX-IMU6_datasheet.pdf"
SAMPLE_CSV = SAMPLES / "failure_notes.csv"

st.set_page_config(page_title="Engineering Document Assistant", page_icon="🛠️", layout="wide")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", ["demo", "groq", "openai", "ollama"], index=0,
                            help="demo replays recorded answers for the sample files; groq has a free tier.")
    model = st.text_input("Model", value=PROVIDERS[provider][0], disabled=provider == "demo")
    key_env = PROVIDERS[provider][1]
    api_key = st.text_input(f"API key ({key_env})", type="password",
                            help="Kept only in this browser session; you can also put it in .env.") if key_env else ""
    st.header("Parameters")
    t_extract = st.slider("Temperature for extraction and labelling", 0.0, 1.5, 0.0, 0.1,
                          help="0 = the same answer every time; data extraction should be repeatable.")
    t_write = st.slider("Temperature for writing (brief, report)", 0.0, 1.5, 0.7, 0.1,
                        help="Higher = more varied wording.")
    top_p = st.slider("top_p", 0.1, 1.0, 1.0, 0.05)
    mt_extract = st.number_input("max_tokens for extraction", 256, 16000, 4096, 256)
    mt_write = st.number_input("max_tokens for writing", 128, 8000, 2048, 64,
                               help="Reasoning models (gpt-oss) also spend these tokens on thinking before they write.")
    if provider == "demo":
        st.info("Demo mode: no API key needed, but only the sample files work.")


def make_client(demo_file: str | None, data: bytes, sample: Path) -> LLMClient:
    answers = None
    if provider == "demo":
        if sha(data) != sha(sample.read_bytes()):
            raise DemoModeError("This is not the sample file.")
        answers = load_demo_answers(SAMPLES / demo_file)
    client = LLMClient(provider=provider, model=model, api_key=api_key, demo_answers=answers,
                       on_wait=lambda seconds, why: st.toast(
                           f"{'Rate limit reached' if why == 'RateLimitError' else 'Provider not answering'}: "
                           f"waiting {seconds:.0f} s, then trying again.", icon="⏳"))
    st.session_state.setdefault("log", [])
    client.log = st.session_state["log"]
    return client


def show_error(e: Exception):
    if isinstance(e, AssistantError):
        st.error(e.user_message)
    else:
        st.error(f"Unexpected error: {type(e).__name__}. The details are below; please report them.")
        with st.expander("Details"):
            st.exception(e)


def progress_bar():
    bar = st.progress(0.0, text="Starting")
    return lambda text, frac: bar.progress(min(max(frac, 0.0), 1.0), text=text)


st.title("🛠️ Engineering Document Assistant")
st.caption("Generative AI for two real engineering jobs: reading component datasheets and analysing factory failure notes.")
tab_ds, tab_q, tab_log, tab_about = st.tabs(["Datasheet Assistant", "Quality Report Analyzer", "Prompts and settings used", "About"])

# ------------------------------------------------------------------------------ datasheet
with tab_ds:
    st.subheader("Datasheet → checked register map, limits, C header and brief")
    col1, col2 = st.columns([3, 1])
    up = col1.file_uploader("Datasheet PDF", type=["pdf"], key="pdf")
    if col2.button("Use the sample datasheet", help="DX-IMU6, a fictional 6-axis IMU"):
        st.session_state["pdf_bytes"] = SAMPLE_PDF.read_bytes()
        st.session_state["pdf_name"] = SAMPLE_PDF.name
    if up is not None:
        st.session_state["pdf_bytes"], st.session_state["pdf_name"] = up.getvalue(), up.name
    pdf_bytes = st.session_state.get("pdf_bytes")
    if pdf_bytes:
        st.write(f"File: **{st.session_state['pdf_name']}** ({len(pdf_bytes) // 1024} KB)")
    if st.button("Analyse datasheet", type="primary", disabled=not pdf_bytes):
        try:
            pages = read_pdf(pdf_bytes)
            client = make_client("demo_answers_datasheet.json", pdf_bytes, SAMPLE_PDF)
            settings = datasheet.DatasheetSettings(
                extract=Params(t_extract, int(mt_extract), top_p), write=Params(t_write, int(mt_write), top_p),
                answer=Params(min(t_extract + 0.2, 1.0), 600, top_p))
            res = datasheet.run(pages, client, settings, progress_bar())
            st.session_state["ds"] = (pages, res, settings)
        except Exception as e:                               # noqa: BLE001 - shown to the user
            show_error(e)

    if "ds" in st.session_state:
        pages, res, settings = st.session_state["ds"]
        for w in res.warnings:
            st.warning(w)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pages", len(pages))
        m2.metric("Registers", len(res.registers.registers))
        m3.metric("Problems found → left", f"{len(res.issues_found)} → {len(res.issues_left)}")
        m4.metric("Scale factors", len(res.scales.scales) if res.scales else 0)
        with st.expander("Page classification (zero-shot)"):
            st.dataframe(pd.DataFrame(sorted(res.page_labels.items()), columns=["page", "category"]), hide_index=True)
        if res.issues_found:
            with st.expander(f"Checks: {len(res.issues_found)} problem(s) found in the first extraction", expanded=True):
                for i in res.issues_found:
                    st.write(("✅ fixed: " if i not in res.issues_left else "⚠️ still open: ") + i)
        st.markdown("#### Register map")
        rows = [{"address": r.address, "name": r.name, "access": r.access, "reset": r.reset,
                 "fields": ", ".join(f"{f.name}[{f.bits}]" for f in r.fields), "description": r.description}
                for r in sorted(res.registers.registers, key=lambda r: int(r.address, 16))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Electrical limits")
            st.dataframe(pd.DataFrame([lim.model_dump() for lim in res.limits.limits]), hide_index=True)
        with c2:
            st.markdown("#### Scale factors (chain-of-thought)")
            if res.scales:
                st.dataframe(pd.DataFrame([s.model_dump() for s in res.scales.scales]), hide_index=True)
        st.markdown("#### Engineering brief")
        st.markdown(res.brief)
        dev = res.registers.device.lower().replace(" ", "_")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ C header", res.header, f"{dev}_regs.h", "text/x-c")
        d2.download_button("⬇️ JSON", datasheet.as_json(res), f"{dev}.json", "application/json")
        d3.download_button("⬇️ Brief (Markdown)", res.brief, f"{dev}_brief.md", "text/markdown")
        with st.expander("C header preview"):
            st.code(res.header, language="c")

        st.markdown("#### Ask the datasheet")
        examples = ["What is the I2C address?", "What pull-up resistors do I need?", "How do I convert the temperature?"]
        pick = st.radio("Examples", examples, horizontal=True, index=None)
        question = st.text_input("Question", value=pick or "")
        if st.button("Ask", disabled=not question):
            try:
                client = make_client("demo_answers_datasheet.json", st.session_state["pdf_bytes"], SAMPLE_PDF)
                answer, used = datasheet.ask(pages, question, client, settings.answer)
                st.markdown(answer)
                st.caption(f"Pages given to the model: {used}")
            except Exception as e:                           # noqa: BLE001
                show_error(e)

# ------------------------------------------------------------------------------ quality
with tab_q:
    st.subheader("Failure notes → labels, groups, root causes, shift-meeting report")
    col1, col2 = st.columns([3, 1])
    upq = col1.file_uploader("Failure notes CSV (note_id, date, shift, station, product, serial_no, note)", type=["csv"], key="csv")
    if col2.button("Use the sample notes", help="58 synthetic notes from 5 days of end-of-line testing"):
        st.session_state["csv_bytes"], st.session_state["csv_name"] = SAMPLE_CSV.read_bytes(), SAMPLE_CSV.name
    if upq is not None:
        st.session_state["csv_bytes"], st.session_state["csv_name"] = upq.getvalue(), upq.name
    csv_bytes = st.session_state.get("csv_bytes")
    if csv_bytes:
        st.write(f"File: **{st.session_state['csv_name']}**")
    if st.button("Analyse notes", type="primary", disabled=not csv_bytes):
        try:
            df = quality.load_notes(csv_bytes)
            client = make_client("demo_answers_quality.json", csv_bytes, SAMPLE_CSV)
            settings = quality.QualitySettings(label=Params(t_extract, int(mt_extract), top_p),
                                               name=Params(min(t_extract + 0.3, 1.5), 800, top_p),
                                               reason=Params(min(t_extract + 0.2, 1.5), int(mt_extract), top_p),
                                               write=Params(t_write, int(mt_write), top_p))
            st.session_state["q"] = quality.run(df, client, settings, progress_bar())
            st.session_state["q_is_sample"] = sha(csv_bytes) == sha(SAMPLE_CSV.read_bytes())
        except Exception as e:                               # noqa: BLE001
            show_error(e)

    if "q" in st.session_state:
        res = st.session_state["q"]
        for w in res.warnings:
            st.warning(w)
        df = res.notes
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Notes", len(df))
        m2.metric("Critical", int((df["severity"] == "critical").sum()))
        m3.metric("Tester suspected", int(df["tester_suspected"].sum()))
        m4.metric("Groups", len(res.clusters))
        if st.session_state.get("q_is_sample"):
            truth = pd.read_csv(SAMPLES / "failure_notes_truth.csv")
            merged = df.merge(truth, on="note_id", suffixes=("", "_true"))
            acc = (merged["failure_mode"] == merged["failure_mode_true"]).mean()
            st.info(f"Labelling accuracy on the sample (known correct answers): {acc:.0%}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Pareto of failure modes")
            chart = res.pareto.assign(mode=[f"{i + 1}. {m}" for i, m in enumerate(res.pareto["failure_mode"])])
            st.bar_chart(chart.set_index("mode")["count"], horizontal=True)   # numbered so the Pareto order stays
            st.dataframe(res.pareto, hide_index=True)
        with c2:
            st.markdown("#### Failures per station")
            st.dataframe(res.by_station)
            st.markdown("#### Groups of similar notes (k-means, named by the model)")
            st.dataframe(pd.DataFrame(res.clusters), hide_index=True)
        st.markdown("#### Root-cause analysis (chain-of-thought)")
        for rc in res.root_causes:
            with st.expander(rc.failure_mode.replace("_", " ")):
                for c in rc.likely_causes:
                    st.write(f"**{c.cause}** ({c.confidence}): {c.evidence}")
                st.write("**Checks:** " + "; ".join(rc.checks))
                st.write("**Containment:** " + rc.containment)
        st.markdown("#### Shift-meeting report")
        st.markdown(res.report)
        out = df.drop(columns=["group"]).copy()
        out["date"] = out["date"].dt.date
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Labelled notes (CSV)", out.to_csv(index=False), "labelled_notes.csv", "text/csv")
        d2.download_button("⬇️ Report (Markdown)", res.report, "shift_report.md", "text/markdown")
        with st.expander("All labelled notes"):
            st.dataframe(out, hide_index=True)

# ------------------------------------------------------------------------------ log
with tab_log:
    log = st.session_state.get("log", [])
    st.write(f"{len(log)} model call(s) in this session. Each shows the prompt technique, the parameters and the exact text sent.")
    if log and st.button("Clear"):
        st.session_state["log"] = []
        st.rerun()
    for rec in reversed(log):
        with st.expander(f"{rec.task} - {rec.technique} - T={rec.params.temperature}, max_tokens={rec.params.max_tokens}, "
                         f"top_p={rec.params.top_p} - {rec.seconds}s, {rec.attempts} attempt(s)"):
            st.markdown("**System prompt**")
            st.code(rec.system, language=None)
            st.markdown("**User prompt**")
            st.code(rec.user[:6000], language=None)
            st.markdown("**Model output**")
            st.code(rec.output[:6000], language=None)

# ------------------------------------------------------------------------------ about
with tab_about:
    st.markdown(Path(__file__).with_name("README.md").read_text(encoding="utf-8").split("## Running it")[0])
