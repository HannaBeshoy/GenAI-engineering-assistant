"""Builds the sample inputs, their hand-checked ground truth, and the recorded answers that
demo mode replays.

    python tools/make_samples.py

samples/DX-IMU6_datasheet.pdf     a datasheet for a FICTIONAL 6-axis IMU (no copyrighted datasheet is shipped)
samples/DX-IMU6_truth.json        the register map, limits and scale factors as printed in that PDF
samples/failure_notes.csv         60 synthetic operator notes from 5 days of end-of-line testing
samples/failure_notes_truth.csv   the correct failure mode of every note (for measuring accuracy)
samples/demo_answers_*.json       what a model answered for these files (used by demo mode and the tests)
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
S = ROOT / "samples"
S.mkdir(exist_ok=True)

# =============================================================== datasheet ground truth
DEVICE = "DX-IMU6"
REGS = [
    ("SMPLRT_DIV", "0x19", "RW", "0x00", "Sample rate divider: rate = 1 kHz / (1 + SMPLRT_DIV)", [("SMPLRT_DIV", "7:0", "Divider value")]),
    ("CONFIG", "0x1A", "RW", "0x00", "Digital low-pass filter configuration", [("EXT_SYNC", "5:3", "External frame sync"), ("DLPF_CFG", "2:0", "Low-pass filter bandwidth")]),
    ("GYRO_CONFIG", "0x1B", "RW", "0x00", "Gyroscope configuration", [("XG_ST", "7", "X gyro self-test"), ("YG_ST", "6", "Y gyro self-test"), ("ZG_ST", "5", "Z gyro self-test"), ("FS_SEL", "4:3", "Gyro full-scale range")]),
    ("ACCEL_CONFIG", "0x1C", "RW", "0x00", "Accelerometer configuration", [("XA_ST", "7", "X accel self-test"), ("YA_ST", "6", "Y accel self-test"), ("ZA_ST", "5", "Z accel self-test"), ("AFS_SEL", "4:3", "Accel full-scale range")]),
    ("INT_ENABLE", "0x38", "RW", "0x00", "Interrupt enable", [("FIFO_OFLOW_EN", "4", "FIFO overflow interrupt enable"), ("DATA_RDY_EN", "0", "Data ready interrupt enable")]),
    ("INT_STATUS", "0x3A", "R", "0x00", "Interrupt status, cleared on read", [("FIFO_OFLOW_INT", "4", "FIFO overflow occurred"), ("DATA_RDY_INT", "0", "New data ready")]),
    ("ACCEL_XOUT_H", "0x3B", "R", "0x00", "Accelerometer X, high byte", []),
    ("ACCEL_XOUT_L", "0x3C", "R", "0x00", "Accelerometer X, low byte", []),
    ("TEMP_OUT_H", "0x41", "R", "0x00", "Temperature, high byte", []),
    ("TEMP_OUT_L", "0x42", "R", "0x00", "Temperature, low byte", []),
    ("GYRO_XOUT_H", "0x43", "R", "0x00", "Gyroscope X, high byte", []),
    ("GYRO_XOUT_L", "0x44", "R", "0x00", "Gyroscope X, low byte", []),
    ("PWR_MGMT_1", "0x6B", "RW", "0x40", "Power management 1", [("DEVICE_RESET", "7", "Reset all registers"), ("SLEEP", "6", "Sleep mode"), ("CYCLE", "5", "Cycle mode"), ("TEMP_DIS", "3", "Disable temperature sensor"), ("CLKSEL", "2:0", "Clock source")]),
    ("WHO_AM_I", "0x75", "R", "0x6C", "Device identity, always reads 0x6C", []),
]
LIMITS = [
    ("Supply voltage", "VDD", -0.3, None, 4.0, "V", "absolute maximum"),
    ("Storage temperature", "TSTG", -40, None, 125, "°C", "absolute maximum"),
    ("Mechanical shock", None, None, None, 10000, "g", "absolute maximum, 0.2 ms"),
    ("Supply voltage", "VDD", 2.375, 3.3, 3.46, "V", None),
    ("I/O supply voltage", "VDDIO", 1.71, 1.8, 3.46, "V", "VDDIO must not exceed VDD"),
    ("Supply current, normal mode", "IDD", None, 3.9, None, "mA", "accel and gyro on"),
    ("Supply current, sleep", "IDD_SLP", None, 5, None, "µA", None),
    ("Operating temperature", "TA", -40, None, 85, "°C", None),
    ("I2C clock frequency", "fSCL", None, None, 400, "kHz", None),
]
SCALES = [("acceleration", "ACCEL_CONFIG", "AFS_SEL", c, f"±{r} g", 32768 / r, "LSB/g") for c, r in enumerate((2, 4, 8, 16))] + \
         [("angular_rate", "GYRO_CONFIG", "FS_SEL", c, f"±{r} °/s", round(32768 / r, 1), "LSB/(°/s)") for c, r in enumerate((250, 500, 1000, 2000))]


def reg_dict(r):
    name, addr, acc, reset, desc, fields = r
    return {"name": name, "address": addr, "access": acc, "reset": reset, "description": desc,
            "fields": [{"name": n, "bits": b, "description": d} for n, b, d in fields]}


TRUTH = {
    "device": DEVICE,
    "registers": [reg_dict(r) for r in REGS],
    "limits": [dict(zip(("parameter", "symbol", "min", "typ", "max", "unit", "conditions"), lim, strict=True)) for lim in LIMITS],
    "scales": [dict(zip(("quantity", "register", "field", "code", "range", "sensitivity", "unit"), s, strict=True)) for s in SCALES],
}


def build_pdf(path: Path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from reportlab import rl_config
    rl_config.invariant = 1          # same bytes on every run (no timestamps)
    st = getSampleStyleSheet()
    H, P = st["Heading1"], st["BodyText"]
    H2 = st["Heading2"]

    def table(rows, widths=None):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4ee")),
                               ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    def fmt(v):
        return "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))

    story = [
        Paragraph(f"{DEVICE} 6-Axis Motion Sensor - Product Datasheet Rev. 1.2", H),
        Paragraph("FICTIONAL DEVICE - sample document for the genai-engineering-assistant project. "
                  "It does not describe a real product.", P), Spacer(1, 4 * mm),
        Paragraph("1 Overview", H2),
        Paragraph(f"The {DEVICE} combines a 3-axis accelerometer and a 3-axis gyroscope with a 16-bit ADC per axis, "
                  "a digital low-pass filter, a temperature sensor and an I2C interface (address 0x68, or 0x69 with AD0 high). "
                  "Typical applications: robot and drone attitude estimation, model-car odometry, vibration monitoring.", P),
        Paragraph("Features: accelerometer ranges ±2, ±4, ±8, ±16 g; gyroscope ranges ±250, ±500, ±1000, ±2000 °/s; "
                  "1 kHz internal sample rate; data-ready interrupt; 3.9 mA in normal mode, 5 µA in sleep.", P),
        PageBreak(),
        Paragraph("2 Pin description", H2),
        table([["Pin", "Name", "Description"], ["1", "VDDIO", "I/O supply"], ["2", "SCL", "I2C clock"],
               ["3", "SDA", "I2C data"], ["4", "AD0", "I2C address bit 0"], ["5", "INT", "Interrupt output, push-pull"],
               ["6", "GND", "Ground"], ["7", "REGOUT", "Regulator filter capacitor, 100 nF to GND"], ["8", "VDD", "Supply"]],
              [15 * mm, 25 * mm, 110 * mm]),
        Paragraph("Place a 100 nF decoupling capacitor close to VDD and to VDDIO. SDA and SCL need pull-up resistors to VDDIO "
                  "(4.7 kΩ for 100 kHz, 2.2 kΩ for 400 kHz).", P),
        PageBreak(),
        Paragraph("3 Electrical characteristics", H2),
        Paragraph("3.1 Absolute maximum ratings (stress beyond these values may cause permanent damage)", P),
        table([["Parameter", "Symbol", "Min", "Max", "Unit", "Notes"]] +
              [[lim[0], lim[1] or "", fmt(lim[2]), fmt(lim[4]), lim[5], (lim[6] or "").replace("absolute maximum", "").strip(", ")]
               for lim in LIMITS[:3]]),
        Spacer(1, 4 * mm),
        Paragraph("3.2 Operating conditions (TA = 25 °C unless noted)", P),
        table([["Parameter", "Symbol", "Min", "Typ", "Max", "Unit", "Conditions"]] +
              [[lim[0], lim[1] or "", fmt(lim[2]), fmt(lim[3]), fmt(lim[4]), lim[5], lim[6] or ""] for lim in LIMITS[3:]]),
        PageBreak(),
        Paragraph("4 I2C interface timing", H2),
        table([["Parameter", "Symbol", "Min", "Max", "Unit"], ["SCL clock frequency", "fSCL", "", "400", "kHz"],
               ["Start condition hold time", "tHD;STA", "0.6", "", "µs"], ["SCL low period", "tLOW", "1.3", "", "µs"],
               ["SCL high period", "tHIGH", "0.6", "", "µs"], ["Data setup time", "tSU;DAT", "100", "", "ns"],
               ["Bus free time between stop and start", "tBUF", "1.3", "", "µs"]]),
        Paragraph("Burst reads auto-increment the register address. Read ACCEL_XOUT_H first: the low byte is latched "
                  "when the high byte is read.", P),
        PageBreak(),
        Paragraph("5 Register map", H2),
        table([["Addr", "Register", "Access", "Reset", "Description"]] +
              [[r[1], r[0], r[2], r[3], r[4]] for r in REGS], [15 * mm, 32 * mm, 15 * mm, 15 * mm, 90 * mm]),
        PageBreak(),
        Paragraph("6 Register descriptions", H2),
    ]
    for r in REGS:
        if not r[5]:
            continue
        story.append(Paragraph(f"6.x {r[0]} (address {r[1]}, {r[2]}, reset value {r[3]})", P))
        story.append(table([["Bits", "Field", "Function"]] + [[b, n, d] for n, b, d in r[5]], [20 * mm, 35 * mm, 100 * mm]))
        if r[0] == "ACCEL_CONFIG":
            story.append(table([["AFS_SEL", "Full-scale range", "Sensitivity"]] +
                               [[str(c), f"±{g} g", f"{32768 // g} LSB/g"] for c, g in enumerate((2, 4, 8, 16))]))
        if r[0] == "GYRO_CONFIG":
            story.append(table([["FS_SEL", "Full-scale range", "Sensitivity"]] +
                               [[str(c), f"±{d} °/s", f"{32768 / d:.1f} LSB/(°/s)"] for c, d in enumerate((250, 500, 1000, 2000))]))
        story.append(Spacer(1, 3 * mm))
    story += [Paragraph("Output data is 16-bit two's complement, high byte first. Temperature in °C = TEMP_OUT / 340 + 36.53.", P),
              PageBreak(), Paragraph("7 Ordering information and revision history", H2),
              table([["Part number", "Package", "Temperature range"], [f"{DEVICE}-QFN", "QFN-24, 4 x 4 mm", "-40 to 85 °C"]]),
              Spacer(1, 3 * mm),
              table([["Revision", "Date", "Change"], ["1.0", "2026-03-01", "First release"], ["1.2", "2026-09-15", "Added I2C timing table"]])]
    SimpleDocTemplate(str(path), pagesize=A4, title=f"{DEVICE} datasheet (fictional)", author="genai-engineering-assistant").build(story)


# =============================================================== failure notes
TEMPLATES = {
    "noise_vibration": [("noise test fail, rumble at high speed", "bearing"), ("vibration velocity {v} mm/s over limit 2.8", "bearing"),
                        ("NOISE NOK - grinding sound when rotating by hand too", "bearing"), ("high frequency noise peak at BPFO, retest fail", "outer ring"),
                        ("noisy, clicking 1x per rev", "bearing")],
    "dimension_out_of_tolerance": [("bore dia {d} over tol (max 20.008)", "inner ring"), ("OD under tolerance, 46.987 measured", "outer ring"),
                                   ("width 14.06 out of spec, tray from grinder 3", "inner ring"), ("bore oval 6 um, limit 4", "inner ring")],
    "seal_damage": [("seal lip cut, visible on one side", "seal"), ("seal not seated, pressed in crooked", "seal"),
                    ("rubber seal torn after press-fit", "seal")],
    "lubrication": [("grease amount low, dry spots on balls", "grease"), ("no grease in one side!", "grease"),
                    ("grease color wrong - looks like the other type", "grease")],
    "contamination": [("metal chip inside bearing", "bearing"), ("dirt particle in raceway after washing", "raceway"),
                      ("hair/fibre found under seal", "seal")],
    "torque_out_of_spec": [("starting torque too high {t} mNm", "bearing"), ("running torque high, stiff rotation", "bearing"),
                           ("actuator holding torque low", "actuator gearbox")],
    "electrical_fault": [("actuator motor no response, connector pin bent", "connector"), ("encoder signal missing ch B", "encoder"),
                         ("motor current 2.1 A over limit at stall", "motor")],
    "test_equipment": [("torque high on fixture B, same part OK on fixture A", "test fixture"), ("tester error E17 - probe not contacting, retest PASS", "test probe"),
                       ("gauge calibration overdue - result not trusted", "gauge"), ("fixture B clamp worn, parts tilt, false fail?", "test fixture")],
    "labeling_handling": [("wrong label - 6205 label on 6204 part", "label"), ("part dropped by operator, scrapped", "handling"),
                          ("serial number not readable", "label")],
}
SEVERITY = {"noise_vibration": "major", "dimension_out_of_tolerance": "major", "seal_damage": "minor", "lubrication": "major",
            "contamination": "major", "torque_out_of_spec": "major", "electrical_fault": "major", "test_equipment": "minor",
            "labeling_handling": "minor"}


def build_notes():
    rnd = random.Random(42)
    plan = ([("noise_vibration", 3, "N", "BRG-6204")] * 12 + [("noise_vibration", s, "", "BRG-6204") for s in (1, 2)] * 2
            + [("dimension_out_of_tolerance", 1, "", "BRG-6204")] * 9 + [("test_equipment", 4, "", "ACT-E40")] * 7
            + [("lubrication", 3, "N", "BRG-6204")] * 4 + [("seal_damage", 2, "", "BRG-6204")] * 4
            + [("torque_out_of_spec", 4, "", "ACT-E40")] * 4 + [("torque_out_of_spec", 1, "", "CRU-200")] * 2
            + [("contamination", 2, "", "CRU-200")] * 3 + [("electrical_fault", 5, "", "ACT-E40")] * 4
            + [("labeling_handling", 5, "", "BRG-6204")] * 3 + [("seal_damage", 2, "", "CRU-200")] * 2)
    rows, truth = [], []
    for i, (mode, station, shift, product) in enumerate(plan, 1):
        text, comp = rnd.choice(TEMPLATES[mode])
        text = text.format(v=round(rnd.uniform(3.0, 4.6), 1), d=round(rnd.uniform(20.009, 20.015), 3), t=rnd.randint(42, 60))
        if rnd.random() < 0.25:
            text += rnd.choice([" - retest also fail", ", 2nd time this shift", " (checked by shift lead)", " pls check"])
        day = 2 + (i * 7) % 5
        rows.append({"note_id": 1000 + i, "date": f"2026-11-0{day}", "shift": shift or rnd.choice("MA"),
                     "station": f"ST{station}", "product": product,
                     "serial_no": f"{product}-2611{day:02d}-{rnd.randint(1, 60):03d}", "note": text})
        crit = mode == "dimension_out_of_tolerance" and "tray" in text
        truth.append({"note_id": 1000 + i, "failure_mode": mode, "component": comp,
                      "severity": "critical" if crit else SEVERITY[mode],
                      "test_equipment_suspected": mode == "test_equipment"})
    rnd.shuffle(rows)
    with open(S / "failure_notes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["date"], r["note_id"])))
    with open(S / "failure_notes_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(truth[0]))
        w.writeheader()
        w.writerows(truth)
    return rows, {t["note_id"]: t for t in truth}


# =============================================================== recorded answers
def demo_datasheet():
    from assistant.pdf import read_pdf
    pages = read_pdf((S / f"{DEVICE}_datasheet.pdf").read_bytes())
    cats = ["overview", "pinout", "electrical", "timing", "register_map", "register_detail", "register_detail", "other"]
    labels = {"pages": [{"page": p.number, "category": cats[min(p.number - 1, len(cats) - 1)]} for p in pages]}
    if len(pages) >= 8:
        labels["pages"][-1]["category"] = "other"
    # the first extraction has two realistic mistakes: a duplicated address and bits written low:high
    first = json.loads(json.dumps(TRUTH["registers"]))
    first[7]["address"] = "0x3B"                                   # ACCEL_XOUT_L copied the H address
    first[1]["fields"][1]["bits"] = "0:2"                          # CONFIG.DLPF_CFG reversed
    extract = "Here is the register map:\n```json\n" + json.dumps({"device": DEVICE, "registers": first}, indent=1, ensure_ascii=False) + "\n```"
    review = json.dumps({"device": DEVICE, "registers": TRUTH["registers"]}, ensure_ascii=False)
    limits = json.dumps({"limits": TRUTH["limits"]}, ensure_ascii=False)
    cot = ("Step 1: ACCEL_CONFIG (0x1C) bits 4:3 are AFS_SEL, the accelerometer full-scale select; GYRO_CONFIG (0x1B) bits 4:3 are FS_SEL.\n"
           "Step 2: AFS_SEL codes 0-3 select ±2, ±4, ±8, ±16 g; FS_SEL codes 0-3 select ±250, ±500, ±1000, ±2000 °/s.\n"
           "Step 3: the datasheet prints the sensitivities: 16384, 8192, 4096, 2048 LSB/g and 131.1, 65.5, 32.8, 16.4 LSB/(°/s). "
           "Check: 32768 / 2 = 16384 and 32768 / 250 = 131.07, consistent with a 16-bit signed output.\n"
           "Step 4: each doubling of the range halves the sensitivity - consistent.\n"
           "<answer>" + json.dumps({"scales": TRUTH["scales"]}, ensure_ascii=False) + "</answer>")
    brief = (f"## What it is\nThe {DEVICE} is a 6-axis IMU (3-axis accelerometer + 3-axis gyroscope, 16-bit) on I2C at 0x68 "
             "(0x69 with AD0 high), with a data-ready interrupt and an on-chip temperature sensor.\n\n"
             "## Key limits\n| Parameter | Min | Typ | Max | Unit |\n|---|---|---|---|---|\n"
             "| Supply voltage VDD | 2.375 | 3.3 | 3.46 | V |\n| VDD absolute maximum | -0.3 | | 4.0 | V |\n"
             "| I/O supply VDDIO | 1.71 | 1.8 | 3.46 | V |\n| Supply current (normal) | | 3.9 | | mA |\n"
             "| Operating temperature | -40 | | 85 | °C |\n| I2C clock | | | 400 | kHz |\n\n"
             "## Wiring checklist\n- [ ] VDD 3.3 V, never above 4.0 V (absolute maximum)\n- [ ] VDDIO not higher than VDD\n"
             "- [ ] 100 nF decoupling at VDD and VDDIO, 100 nF from REGOUT to GND\n- [ ] SDA/SCL pull-ups to VDDIO: 2.2 kΩ for 400 kHz\n"
             "- [ ] AD0 tied low (address 0x68) or high (0x69), never floating\n\n"
             "## Driver start-up sequence\n1. Read WHO_AM_I (0x75): must be 0x6C.\n2. Write PWR_MGMT_1 (0x6B) = 0x00 to leave sleep (reset value 0x40 = SLEEP set).\n"
             "3. Write SMPLRT_DIV (0x19) = 9 for 100 Hz, CONFIG (0x1A) DLPF_CFG for the filter.\n"
             "4. Write ACCEL_CONFIG (0x1C) AFS_SEL and GYRO_CONFIG (0x1B) FS_SEL; divide raw values by 16384 LSB/g and 131.1 LSB/(°/s) at the smallest ranges.\n"
             "5. Enable DATA_RDY_EN in INT_ENABLE (0x38); read 14 bytes from 0x3B in one burst.\n\n"
             "## Gotchas\n- The part starts in sleep mode: nothing changes until SLEEP is cleared.\n- Read the high byte first; the low byte is latched then.\n"
             "- INT_STATUS clears on read: read it once per interrupt.\n- Mechanical shock above 10000 g can damage it: do not drop the board.")
    qa = {"What is the I2C address": "The I2C address is 0x68, or 0x69 when the AD0 pin is high [p. 1]. Tie AD0 to a defined level [p. 2].",
          "What pull-up resistors": "Use pull-ups to VDDIO: 4.7 kΩ for 100 kHz or 2.2 kΩ for 400 kHz [p. 2]. The maximum SCL clock is 400 kHz [p. 4].",
          "How do I convert the temperature": "Temperature in °C = TEMP_OUT / 340 + 36.53, where TEMP_OUT is the signed 16-bit value from TEMP_OUT_H and TEMP_OUT_L [p. 6]."}
    return {"classify_pages": [json.dumps(labels)], "extract_registers": [extract], "review_registers": [review],
            "extract_limits": [limits], "derive_scales": [cot], "engineering_brief": [brief], "ask_datasheet": qa}


def demo_quality(rows, truth):
    import pandas as pd
    from assistant.quality import BATCH, _cluster, load_notes
    df = load_notes((S / "failure_notes.csv").read_bytes())
    batches, given = [], []
    for i in range(0, len(df), BATCH):
        part = df.iloc[i:i + BATCH]
        labs = [dict(truth[n]) for n in part["note_id"]]
        for lab in labs:                                            # one realistic disagreement
            if lab["note_id"] == part["note_id"].iloc[0] and lab["failure_mode"] == "lubrication":
                lab["failure_mode"] = "noise_vibration"
        given += labs
        text = json.dumps({"labels": labs}, indent=1)
        if i == 0:
            text = "```json\n" + text.replace("\n]", ",\n]") + "\n```"   # trailing comma + fence: the parser must cope
        batches.append(text)
    groups, keywords, _ = _cluster(df["note"].tolist())
    df["group"] = groups
    names = []
    for g, kw in enumerate(keywords):
        modes = pd.Series([truth[n]["failure_mode"] for n in df[df["group"] == g]["note_id"]]).value_counts()
        top = modes.index[0].replace("_", " ")
        names.append({"cluster": g, "name": f"{top.capitalize()} notes"[:40],
                      "description": f"Notes mostly about {top} (keywords: {', '.join(kw[:3])})."})
    counts = pd.Series([lab["failure_mode"] for lab in given]).value_counts()   # what the model said
    top_modes = [m for m in counts.index if m != "other"][:3]
    rc_text = {
        "noise_vibration": ("Step 1: 13 of 17 noise failures come from station ST3 on the night shift, all BRG-6204; the other 4 are spread over ST1 and ST2.\n"
                            "Step 2: a pattern tied to one station and one shift points to that line's setup or material rather than the design. "
                            "ST3 night shift also has all 3 lubrication notes (low grease, dry spots).\n"
                            "Step 3: dry or under-greased bearings are noisy, so a greasing problem on ST3 at night explains both modes; "
                            "a worn noise-test fixture would also show on day shifts, which it does not.\n"
                            "Step 4: check the grease dosing unit on ST3, the night-shift grease batch and the setup sheet.\n<answer>",
                            {"failure_mode": "noise_vibration", "likely_causes": [
                                {"cause": "Under-greasing by the ST3 grease dosing unit on night shift", "evidence": "13/17 noise notes and 3/3 lubrication notes are ST3 night shift", "confidence": "high"},
                                {"cause": "Different grease batch loaded for night shift", "evidence": "one note says the grease colour looks wrong", "confidence": "medium"},
                                {"cause": "Raceway damage from upstream grinding", "evidence": "noise also seen on ST1/ST2, but only 4 cases", "confidence": "low"}],
                             "checks": ["Weigh grease fill of 5 night-shift parts from ST3", "Compare grease batch numbers day vs night", "Run the noise test on 3 ST3 parts at ST1"],
                             "containment": "100% noise retest of ST3 night-shift output since 2026-11-02; hold the current grease batch."}),
        "dimension_out_of_tolerance": ("Step 1: all 9 dimension failures are BRG-6204 at ST1, several mention grinder 3 and a whole tray.\n"
                                       "Step 2: oversize bore and width point to the grinding process or its gauge.\n"
                                       "Step 3: values drift just above the limit, typical of wheel wear without compensation.\n"
                                       "Step 4: check the grinder 3 compensation and the in-process gauge master.\n<answer>",
                                       {"failure_mode": "dimension_out_of_tolerance", "likely_causes": [
                                           {"cause": "Grinding wheel wear on grinder 3 not compensated", "evidence": "bore values 20.009-20.015 mm, just over 20.008", "confidence": "high"},
                                           {"cause": "In-process gauge out of calibration", "evidence": "parts pass the grinder gauge but fail at ST1", "confidence": "medium"}],
                                        "checks": ["Measure 10 parts from grinder 3 on the CMM", "Check the gauge master and the last calibration date", "Review the dressing interval"],
                                        "containment": "Quarantine trays from grinder 3 since the first failure; sort by 100% bore measurement."}),
        "test_equipment": ("Step 1: 7 notes from ST4 on ACT-E40 blame fixture B or the probe; retests often pass.\n"
                           "Step 2: failures that disappear on retest or on fixture A point to the tester, not the part.\n"
                           "Step 3: a worn clamp tilting the part explains high torque readings.\n"
                           "Step 4: repair fixture B and do a gauge R&R before trusting its results.\n<answer>",
                           {"failure_mode": "test_equipment", "likely_causes": [
                               {"cause": "Worn clamp on ST4 fixture B tilts the part", "evidence": "same parts pass on fixture A; note mentions the worn clamp", "confidence": "high"},
                               {"cause": "Gauge calibration overdue", "evidence": "one note says calibration is overdue", "confidence": "medium"}],
                            "checks": ["Gauge R&R study on fixture B", "Replace the fixture B clamp", "Check calibration labels on all ST4 gauges"],
                            "containment": "Run ACT-E40 only on fixture A; retest all fixture-B fails on fixture A before scrapping."}),
    }
    root = []
    for m in top_modes:
        pre, data = rc_text.get(m, ("Step 1: summary.\n<answer>", {"failure_mode": m, "likely_causes": [{"cause": "unknown", "evidence": "few notes", "confidence": "low"}], "checks": ["collect more data"], "containment": "none"}))
        root.append(pre + json.dumps(data, ensure_ascii=False) + "</answer>")
    total = len(truth)
    report = (f"## Summary\n- {total} failure notes from 2 to 6 November; the top three problems cause "
              f"{round(100 * counts[top_modes].sum() / total)}% of them.\n"
              f"- Noise/vibration ({counts['noise_vibration']}) is concentrated on **ST3 night shift**, together with all lubrication findings.\n"
              f"- {counts['test_equipment']} failures on ST4 are probably **false fails** from fixture B.\n\n"
              "## Top problems\n| Failure mode | Count | Share | Trend |\n|---|---|---|---|\n"
              + "".join(f"| {m.replace('_', ' ')} | {counts[m]} | {round(100 * counts[m] / total)}% | stable |\n" for m in counts.index[:5])
              + "\n## Root-cause status\n- Noise: grease dosing on ST3 (high confidence) - verify today.\n"
              "- Dimensions: grinder 3 wheel wear (high) - CMM check.\n- Test equipment: fixture B clamp (high) - repair.\n\n"
              "## Actions for today\n- [ ] Weigh grease fill on 5 ST3 parts - process engineer\n"
              "- [ ] 100% noise retest of ST3 night output - quality technician\n- [ ] Quarantine grinder 3 trays and sort - shift lead\n"
              "- [ ] Move ACT-E40 testing to fixture A; repair fixture B - maintenance")
    return {"classify_notes": batches, "name_clusters": [json.dumps({"clusters": names})],
            "root_cause": root, "shift_report": [report]}


if __name__ == "__main__":
    build_pdf(S / f"{DEVICE}_datasheet.pdf")
    (S / f"{DEVICE}_truth.json").write_text(json.dumps(TRUTH, indent=2, ensure_ascii=False), encoding="utf-8")
    rows, truth = build_notes()
    (S / "demo_answers_datasheet.json").write_text(json.dumps(demo_datasheet(), indent=1, ensure_ascii=False), encoding="utf-8")
    (S / "demo_answers_quality.json").write_text(json.dumps(demo_quality(rows, truth), indent=1, ensure_ascii=False), encoding="utf-8")
    print("samples written:", sorted(p.name for p in S.iterdir()))
