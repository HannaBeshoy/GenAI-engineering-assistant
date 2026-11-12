"""Tool 2: operator failure notes (CSV) -> labelled, grouped, root-caused, shift-meeting report.

Steps:
  1 load + check the CSV (code)
  2 classify_notes   few-shot, batches of 15    failure mode, component, severity, tester suspected
  3 grouping (code)  TF-IDF + k-means, k chosen by silhouette score
  4 name_clusters    zero-shot                  a name and a description for each group
  5 statistics (code) Pareto, per station / product / shift, trend per day
  6 root_cause       chain-of-thought + persona for the top failure modes
  7 shift_report     persona, temperature 0.7   Markdown for the 6 a.m. meeting
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from . import prompts as P
from .errors import BadModelOutput, InputDataError
from .llm import LLMClient, Params
from .schemas import FAILURE_MODES, ClusterNames, NoteLabels, RootCause
from .structured import ask_structured

REQUIRED = ["note_id", "date", "shift", "station", "product", "serial_no", "note"]
BATCH = 15
MAX_ROWS = 500


@dataclass
class QualitySettings:
    label: Params = field(default_factory=lambda: Params(temperature=0.0, max_tokens=2048))
    name: Params = field(default_factory=lambda: Params(temperature=0.3, max_tokens=800))
    reason: Params = field(default_factory=lambda: Params(temperature=0.2, max_tokens=1500))
    write: Params = field(default_factory=lambda: Params(temperature=0.7, max_tokens=2048))
    top_modes: int = 3


@dataclass
class QualityResult:
    notes: pd.DataFrame = None
    clusters: list = field(default_factory=list)
    pareto: pd.DataFrame = None
    by_station: pd.DataFrame = None
    root_causes: list = field(default_factory=list)
    report: str = ""
    warnings: list = field(default_factory=list)


def load_notes(data: bytes) -> pd.DataFrame:
    if not data:
        raise InputDataError("The CSV file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", dtype=str)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        raise InputDataError(f"The file could not be read as CSV ({type(e).__name__}).") from e
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputDataError(f"Missing column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED)}.")
    df = df[REQUIRED].copy()
    df["note"] = df["note"].fillna("").str.strip()
    df = df[df["note"] != ""]
    if df.empty:
        raise InputDataError("No rows with a non-empty note.")
    if len(df) > MAX_ROWS:
        raise InputDataError(f"{len(df)} notes; the limit is {MAX_ROWS} per run.")
    try:
        df["note_id"] = df["note_id"].astype(int)
    except ValueError:
        raise InputDataError("note_id must be a whole number in every row.") from None
    if df["note_id"].duplicated().any():
        raise InputDataError("note_id values must be unique.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise InputDataError("Some dates could not be read; use YYYY-MM-DD.")
    return df.reset_index(drop=True)


def _cluster(texts: list) -> tuple:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(texts)
    best = (None, -1.0, 1)
    for k in range(3, min(8, len(texts) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        score = silhouette_score(X, km.labels_)
        if score > best[1]:
            best = (km, score, k)
    km = best[0]
    if km is None:
        return [0] * len(texts), [[]], 0.0
    terms = vec.get_feature_names_out()
    keywords = [[terms[i] for i in centre.argsort()[::-1][:6]] for centre in km.cluster_centers_]
    return list(km.labels_), keywords, round(best[1], 3)


def run(df: pd.DataFrame, client: LLMClient, settings: QualitySettings,
        progress: Callable[[str, float], None] = lambda s, f: None) -> QualityResult:
    res = QualityResult()
    s = settings
    df = df.copy()

    # 2 classify --------------------------------------------------------------------------
    labels = {}
    batches = [df.iloc[i:i + BATCH] for i in range(0, len(df), BATCH)]
    for b, batch in enumerate(batches, 1):
        progress(f"Labelling notes (few-shot), batch {b}/{len(batches)}", 0.05 + 0.4 * b / len(batches))
        text = "\n".join(f'note {r.note_id}: "{r.note}"' for r in batch.itertuples())
        out = ask_structured(client, task="classify_notes", technique=P.TECHNIQUES["classify_notes"],
                             system=P.CLASSIFY_NOTES_SYSTEM,
                             user=P.CLASSIFY_NOTES_USER.format(modes=", ".join(FAILURE_MODES), notes=text),
                             schema=NoteLabels, params=s.label)
        wanted = set(batch["note_id"])
        for lab in out.labels:
            if lab.note_id in wanted:
                labels[lab.note_id] = lab
        lost = wanted - {lab.note_id for lab in out.labels}
        if lost:
            res.warnings.append(f"The model skipped note(s) {sorted(lost)}; they are marked 'other'.")
    df["failure_mode"] = [labels[i].failure_mode if i in labels else "other" for i in df["note_id"]]
    df["component"] = [labels[i].component if i in labels else "" for i in df["note_id"]]
    df["severity"] = [labels[i].severity if i in labels else "major" for i in df["note_id"]]
    df["tester_suspected"] = [labels[i].test_equipment_suspected if i in labels else False for i in df["note_id"]]

    # 3-4 group and name --------------------------------------------------------------------
    progress("Grouping similar notes (TF-IDF + k-means)", 0.5)
    groups, keywords, sil = _cluster(df["note"].tolist())
    df["group"] = groups
    blocks = []
    for g, kw in enumerate(keywords):
        examples = df[df["group"] == g]["note"].head(4).tolist()
        blocks.append(f"group {g} ({(df['group'] == g).sum()} notes, keywords: {', '.join(kw)}):\n"
                      + "\n".join(f"- {e}" for e in examples))
    progress("Naming the groups (zero-shot)", 0.55)
    try:
        names = ask_structured(client, task="name_clusters", technique=P.TECHNIQUES["name_clusters"],
                               system=P.NAME_CLUSTERS_SYSTEM, user=P.NAME_CLUSTERS_USER.format(groups="\n\n".join(blocks)),
                               schema=ClusterNames, params=s.name)
        named = {c.cluster: c for c in names.clusters}
    except BadModelOutput as e:
        res.warnings.append(f"Groups are not named: {e}")
        named = {}
    for g, kw in enumerate(keywords):
        c = named.get(g)
        res.clusters.append({"group": g, "name": c.name if c else f"group {g}",
                             "description": c.description if c else "", "keywords": ", ".join(kw),
                             "notes": int((df["group"] == g).sum())})
    df["group_name"] = [res.clusters[g]["name"] for g in df["group"]]

    # 5 statistics ----------------------------------------------------------------------------
    progress("Statistics", 0.6)
    total = len(df)
    pareto = df["failure_mode"].value_counts().rename_axis("failure_mode").reset_index(name="count")
    pareto["share_%"] = (100 * pareto["count"] / total).round(1)
    pareto["cumulative_%"] = (100 * pareto["count"].cumsum() / total).round(1)
    res.pareto = pareto
    res.by_station = pd.crosstab(df["station"], df["failure_mode"])

    # 6 root cause for the top modes ----------------------------------------------------------
    top = [m for m in pareto["failure_mode"] if m != "other"][:s.top_modes]
    for i, mode in enumerate(top, 1):
        progress(f"Root-cause reasoning for '{mode}' (chain-of-thought)", 0.6 + 0.25 * i / max(len(top), 1))
        sub = df[df["failure_mode"] == mode]
        stats = {
            "by_station": sub["station"].value_counts().to_dict(),
            "by_product": sub["product"].value_counts().to_dict(),
            "by_shift": sub["shift"].value_counts().to_dict(),
            "by_day": {str(k.date()): int(v) for k, v in sub.groupby(sub["date"].dt.normalize()).size().items()},
            "tester_suspected": int(sub["tester_suspected"].sum()),
        }
        notes = "\n".join(f"- {r.date.date()} shift {r.shift} station {r.station} {r.product}: {r.note}"
                          for r in sub.head(25).itertuples())
        try:
            rc = ask_structured(client, task="root_cause", technique=P.TECHNIQUES["root_cause"],
                                system=P.ROOT_CAUSE_SYSTEM,
                                user=P.ROOT_CAUSE_USER.format(mode=mode, count=len(sub), total=total,
                                                              stats=json.dumps(stats, indent=1), notes=notes),
                                schema=RootCause, params=s.reason)
            res.root_causes.append(rc)
        except BadModelOutput as e:
            res.warnings.append(f"No root-cause analysis for {mode}: {e}")

    # 7 report --------------------------------------------------------------------------------
    progress("Writing the shift-meeting report (persona)", 0.9)
    period = f"{df['date'].min().date()} to {df['date'].max().date()}"
    totals = {"notes": total, "critical": int((df["severity"] == "critical").sum()),
              "tester_suspected": int(df["tester_suspected"].sum()), "groups": len(res.clusters),
              "silhouette": sil}
    res.report = client.complete(
        "shift_report", P.TECHNIQUES["shift_report"], P.SHIFT_REPORT_SYSTEM,
        P.SHIFT_REPORT_USER.format(period=period, totals=json.dumps(totals),
                                   pareto=pareto.to_csv(index=False),
                                   root_causes=json.dumps([r.model_dump() for r in res.root_causes])[:6000]),
        s.write)
    res.notes = df
    progress("Done", 1.0)
    return res
