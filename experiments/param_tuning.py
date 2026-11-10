"""Parameter-tuning experiment: how temperature and max_tokens change the quality of the
extraction and labelling steps, measured against the samples' known correct answers.

    python experiments/param_tuning.py --provider groq                  # needs GROQ_API_KEY
    python experiments/param_tuning.py --provider groq --runs 3 --temps 0 0.4 0.8 1.2

For every temperature it runs the register extraction and the note labelling `--runs` times and
records: valid-JSON rate (before repair), register F1, field F1, labelling accuracy, and how often
two runs agree with each other. A second part lowers max_tokens until the answer is cut off, to
show the truncation handling. Results: experiments/results/param_tuning.csv and .md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from assistant import prompts as P  # noqa: E402
from assistant.errors import AssistantError, TruncatedOutput  # noqa: E402
from assistant.llm import LLMClient, Params  # noqa: E402
from assistant.pdf import read_pdf  # noqa: E402
from assistant.quality import load_notes  # noqa: E402
from assistant.schemas import FAILURE_MODES, NoteLabels, RegisterMap  # noqa: E402
from assistant.structured import extract_json  # noqa: E402

S = ROOT / "samples"
OUT = ROOT / "experiments" / "results"


def f1(found: set, truth: set) -> float:
    if not found or not truth:
        return 0.0
    tp = len(found & truth)
    p, r = tp / len(found), tp / len(truth)
    return 0.0 if tp == 0 else round(2 * p * r / (p + r), 3)


def registers_once(client, text, params):
    raw = client.complete("extract_registers", "few-shot", P.EXTRACT_REGISTERS_SYSTEM,
                          P.EXTRACT_REGISTERS_USER.format(text=text), params)
    try:
        rmap = RegisterMap.model_validate(extract_json(raw))
    except Exception:                                    # noqa: BLE001 - counted as invalid
        return None
    return rmap


def labels_once(client, notes, params):
    text = "\n".join(f'note {r.note_id}: "{r.note}"' for r in notes.itertuples())
    raw = client.complete("classify_notes", "few-shot", P.CLASSIFY_NOTES_SYSTEM,
                          P.CLASSIFY_NOTES_USER.format(modes=", ".join(FAILURE_MODES), notes=text), params)
    try:
        return {lab.note_id: lab.failure_mode for lab in NoteLabels.model_validate(extract_json(raw)).labels}
    except Exception:                                    # noqa: BLE001
        return None


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", default="")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.4, 0.8, 1.2])
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between calls (free-tier rate limits)")
    a = ap.parse_args()

    client = LLMClient(a.provider, a.model)
    truth = json.loads((S / "DX-IMU6_truth.json").read_text(encoding="utf-8"))
    t_regs = {(r["name"], r["address"]) for r in truth["registers"]}
    t_fields = {(r["name"], f["name"], f["bits"]) for r in truth["registers"] for f in r["fields"]}
    pages = read_pdf((S / "DX-IMU6_datasheet.pdf").read_bytes())
    reg_text = "\n\n".join(f"--- page {p.number} ---\n{p.text}" for p in pages if p.number in (5, 6, 7))
    notes = load_notes((S / "failure_notes.csv").read_bytes()).head(30)
    t_labels = {int(r["note_id"]): r["failure_mode"] for r in csv.DictReader(open(S / "failure_notes_truth.csv"))}

    rows = []
    for t in a.temps:
        params = Params(temperature=t, max_tokens=4096)
        reg_runs, lab_runs = [], []
        for run in range(a.runs):
            start = time.monotonic()
            try:
                rmap = registers_once(client, reg_text, params)
                time.sleep(a.pause)
                labs = labels_once(client, notes, params)
            except AssistantError as e:
                print(f"T={t} run {run}: {e.user_message}")
                rmap, labs = None, None
            time.sleep(a.pause)
            reg_runs.append(rmap)
            lab_runs.append(labs)
            regs = {(r.name, r.address) for r in rmap.registers} if rmap else set()
            fields = {(r.name, f.name, f.bits) for r in rmap.registers for f in r.fields} if rmap else set()
            acc = (sum(labs.get(i) == m for i, m in t_labels.items() if i in set(notes.note_id)) / len(notes)) if labs else 0.0
            rows.append({"temperature": t, "run": run, "registers_valid_json": rmap is not None,
                         "register_f1": f1(regs, t_regs), "field_f1": f1(fields, t_fields),
                         "labels_valid_json": labs is not None, "label_accuracy": round(acc, 3),
                         "seconds": round(time.monotonic() - start, 1)})
            print(rows[-1])
        same = [lab_runs[i] == lab_runs[0] for i in range(1, len(lab_runs)) if lab_runs[0] is not None]
        rows[-1]["runs_agree"] = all(same) if same else ""

    trunc = []
    for mt in (2048, 512, 128):
        try:
            client.complete("extract_registers", "few-shot", P.EXTRACT_REGISTERS_SYSTEM,
                            P.EXTRACT_REGISTERS_USER.format(text=reg_text), Params(0.0, mt))
            trunc.append((mt, "complete answer"))
        except TruncatedOutput as e:
            trunc.append((mt, f"TruncatedOutput: {e}"))
        time.sleep(a.pause)

    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) + ["runs_agree"]
    with open(OUT / "param_tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = [f"# Parameter tuning ({a.provider}, {client.model}, {a.runs} runs per temperature)", "",
          "| temperature | valid JSON (registers) | register F1 | field F1 | valid JSON (labels) | label accuracy |",
          "| --- | --- | --- | --- | --- | --- |"]
    for t in a.temps:
        rs = [r for r in rows if r["temperature"] == t]
        avg = lambda k, rs=rs: round(sum(float(r[k]) for r in rs) / len(rs), 3)       # noqa: E731
        md.append(f"| {t} | {sum(r['registers_valid_json'] for r in rs)}/{len(rs)} | {avg('register_f1')} | {avg('field_f1')} "
                  f"| {sum(r['labels_valid_json'] for r in rs)}/{len(rs)} | {avg('label_accuracy')} |")
    md += ["", "| max_tokens | result |", "| --- | --- |"] + [f"| {m} | {r} |" for m, r in trunc]
    (OUT / "param_tuning.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
