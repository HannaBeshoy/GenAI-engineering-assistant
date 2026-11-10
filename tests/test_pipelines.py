import json

import pandas as pd
import pytest
from conftest import SAMPLES

from assistant import datasheet, quality
from assistant.errors import DemoModeError, InputDataError
from assistant.llm import LLMClient, Params, load_demo_answers
from assistant.pdf import read_pdf


@pytest.fixture(scope="module")
def ds_result():
    c = LLMClient("demo", demo_answers=load_demo_answers(SAMPLES / "demo_answers_datasheet.json"))
    pages = read_pdf((SAMPLES / "DX-IMU6_datasheet.pdf").read_bytes())
    return pages, datasheet.run(pages, c, datasheet.DatasheetSettings()), c


def test_datasheet_matches_ground_truth(ds_result):
    _, res, _ = ds_result
    truth = json.loads((SAMPLES / "DX-IMU6_truth.json").read_text(encoding="utf-8"))
    out = json.loads(datasheet.as_json(res))
    assert out["registers"] == truth["registers"]
    assert out["limits"] == truth["limits"]
    assert out["scales"] == truth["scales"]


def test_checks_found_and_review_fixed_the_mistakes(ds_result):
    _, res, c = ds_result
    assert len(res.issues_found) == 2 and res.issues_left == []
    assert [r.task for r in c.log] == ["classify_pages", "extract_registers", "extract_limits",
                                       "review_registers", "derive_scales", "engineering_brief"]
    assert {r.technique for r in c.log} >= {"zero-shot", "few-shot", "chain-of-thought", "persona"}


def test_header(ds_result):
    _, res, _ = ds_result
    h = res.header
    assert "#define DX_IMU6_PWR_MGMT_1_REG        0x6B" in h
    assert "#define DX_IMU6_ACCEL_CONFIG_AFS_SEL_MSK   0x18u" in h
    assert "#define DX_IMU6_SENS_ACCELERATION_2G   16384f" in h
    assert h.count("#ifndef") == 1 and h.rstrip().endswith("*/")


def test_question_uses_relevant_pages(ds_result):
    pages, _, _ = ds_result
    c = LLMClient("demo", demo_answers=load_demo_answers(SAMPLES / "demo_answers_datasheet.json"))
    answer, used = datasheet.ask(pages, "What is the I2C address?", c, Params(0.2, 600))
    assert "0x68" in answer and 1 in used
    with pytest.raises(DemoModeError):
        datasheet.ask(pages, "What is the maximum supply voltage?", c, Params())


@pytest.fixture(scope="module")
def q_result():
    c = LLMClient("demo", demo_answers=load_demo_answers(SAMPLES / "demo_answers_quality.json"))
    df = quality.load_notes((SAMPLES / "failure_notes.csv").read_bytes())
    return quality.run(df, c, quality.QualitySettings()), c


def test_quality_labels_and_accuracy(q_result):
    res, _ = q_result
    truth = pd.read_csv(SAMPLES / "failure_notes_truth.csv")
    m = res.notes.merge(truth, on="note_id", suffixes=("", "_true"))
    assert len(m) == 58
    assert (m["failure_mode"] == m["failure_mode_true"]).mean() > 0.95


def test_quality_statistics_and_root_causes(q_result):
    res, c = q_result
    assert res.pareto.iloc[0]["failure_mode"] == "noise_vibration"
    assert res.pareto["cumulative_%"].iloc[-1] == 100.0
    assert [r.failure_mode for r in res.root_causes] == ["noise_vibration", "dimension_out_of_tolerance", "test_equipment"]
    assert "ST3" in res.root_causes[0].likely_causes[0].evidence
    assert 3 <= len(res.clusters) <= 8 and all(cl["name"] for cl in res.clusters)
    assert "## Actions for today" in res.report


def test_skipped_notes_are_reported():
    df = quality.load_notes((SAMPLES / "failure_notes.csv").read_bytes()).head(3)
    first = int(df.note_id.iloc[0])
    answers = {"classify_notes": [json.dumps({"labels": [{"note_id": first, "failure_mode": "seal_damage", "component": "seal",
                                                           "severity": "minor"}]})],
               "name_clusters": ['{"clusters": []}'], "root_cause": ["<answer>{}</answer>"], "root_cause:repair": ["no"],
               "shift_report": ["report"]}
    res = quality.run(df, LLMClient("demo", demo_answers=answers), quality.QualitySettings(top_modes=1))
    assert any("skipped" in w for w in res.warnings)
    assert (res.notes["failure_mode"] == "other").sum() == 2


@pytest.mark.parametrize("csv,msg", [
    (b"", "empty"),
    (b"note_id,date,note\n1,2026-11-02,x\n", "Missing column"),
    (b"note_id,date,shift,station,product,serial_no,note\n1,2026-11-02,A,ST1,P,S,x\n1,2026-11-02,A,ST1,P,S,y\n", "unique"),
    (b"note_id,date,shift,station,product,serial_no,note\n1,yesterday,A,ST1,P,S,x\n", "dates"),
    (b"note_id,date,shift,station,product,serial_no,note\n1,2026-11-02,A,ST1,P,S,\n", "non-empty"),
])
def test_bad_csv(csv, msg):
    with pytest.raises(InputDataError, match=msg):
        quality.load_notes(csv)
