"""Every prompt of the app, with the prompt-engineering technique it uses.

| Step | Technique | Temperature |
| --- | --- | --- |
| classify_pages | zero-shot classification | 0.0 |
| extract_registers | few-shot (one worked example) | 0.0 |
| extract_limits | few-shot | 0.0 |
| derive_scales | chain-of-thought, answer in <answer> tags | 0.0 |
| *:repair | self-correction with validator feedback | 0.0 |
| review_registers | persona (senior embedded engineer) + validator findings | 0.0 |
| engineering_brief | persona + higher temperature for readable prose | 0.7 |
| ask_datasheet | retrieval-grounded zero-shot with page citations | 0.2 |
| classify_notes | few-shot (three labelled examples) | 0.0 |
| name_clusters | zero-shot topic naming | 0.3 |
| root_cause | chain-of-thought + persona (quality engineer, 8D style) | 0.2 |
| shift_report | persona + higher temperature | 0.7 |
"""

TECHNIQUES = {
    "classify_pages": "zero-shot",
    "extract_registers": "few-shot",
    "extract_limits": "few-shot",
    "derive_scales": "chain-of-thought",
    "review_registers": "persona + validator feedback",
    "engineering_brief": "persona",
    "ask_datasheet": "retrieval-grounded zero-shot",
    "classify_notes": "few-shot",
    "name_clusters": "zero-shot",
    "root_cause": "chain-of-thought + persona",
    "shift_report": "persona",
}

JSON_ONLY = "Answer with JSON only: no explanation, no code fences."

# ------------------------------------------------------------------ datasheet

CLASSIFY_PAGES_SYSTEM = "You classify pages of electronic component datasheets. " + JSON_ONLY

CLASSIFY_PAGES_USER = """Classify each page into exactly one category:
- overview: product description, features, applications
- pinout: pin names, pin numbers, package drawing
- electrical: absolute maximum ratings, electrical characteristics, operating conditions
- timing: timing diagrams, interface timing tables (I2C/SPI)
- register_map: a table listing many registers with addresses
- register_detail: the description of individual registers and their bit fields
- other: anything else (ordering information, revision history, ...)

Return {{"pages": [{{"page": <number>, "category": "<category>"}}, ...]}} with one entry per page.

Pages (first 700 characters of each):
{pages}"""

EXTRACT_REGISTERS_SYSTEM = ("You extract register maps from datasheets for embedded C drivers. "
                            "Copy names, addresses and reset values exactly as printed; never invent registers. " + JSON_ONLY)

EXTRACT_REGISTERS_USER = """Extract every register with its bit fields.

Example (from a different device):
Text:
"0x0F WHO_AM_I R 0x33 Device identification. 0x20 CTRL_REG1 RW 0x07 Bits 7:4 ODR output data rate; bit 3 LPEN low-power enable; bits 2:0 XYZEN axis enable."
JSON:
{{"device": "EXAMPLE-ACC", "registers": [
  {{"name": "WHO_AM_I", "address": "0x0F", "access": "R", "reset": "0x33", "description": "Device identification", "fields": []}},
  {{"name": "CTRL_REG1", "address": "0x20", "access": "RW", "reset": "0x07", "description": "Control register 1", "fields": [
    {{"name": "ODR", "bits": "7:4", "description": "Output data rate"}},
    {{"name": "LPEN", "bits": "3", "description": "Low-power enable"}},
    {{"name": "XYZEN", "bits": "2:0", "description": "Axis enable"}}]}}]}}

Rules: address and reset as 0x followed by two hex digits; access is R, W or RW; bits as "7:5" or "3".

Now the real datasheet text:
{text}"""

EXTRACT_LIMITS_USER = """Extract the electrical limits as a list.

Example:
Text: "Supply voltage VDD 1.71 1.8 3.6 V. Operating temperature TA -40 85 °C."
JSON:
{{"limits": [
  {{"parameter": "Supply voltage", "symbol": "VDD", "min": 1.71, "typ": 1.8, "max": 3.6, "unit": "V", "conditions": null}},
  {{"parameter": "Operating temperature", "symbol": "TA", "min": -40, "typ": null, "max": 85, "unit": "°C", "conditions": null}}]}}

Use null for an empty cell. Keep absolute maximum ratings, marking them in "conditions" as "absolute maximum".

Datasheet text:
{text}"""

DERIVE_SCALES_SYSTEM = "You are an embedded engineer working out sensor scale factors from a datasheet."

DERIVE_SCALES_USER = """For every full-scale range setting of the sensor, find the register field that selects it,
the code written to that field, and the sensitivity in LSB per unit.

Think step by step first: (1) find the configuration register and field that selects the range,
(2) list each code and its range, (3) find or calculate the sensitivity for each range
(for a 16-bit signed output, sensitivity = 32768 / full-scale value, if the datasheet does not print it),
(4) check that the sensitivity halves when the range doubles.

Then give the final result between <answer> and </answer> as
{{"scales": [{{"quantity": "acceleration|angular_rate|temperature|other", "register": "...", "field": "...",
"code": 0, "range": "±2 g", "sensitivity": 16384, "unit": "LSB/g"}}, ...]}}

Datasheet text:
{text}"""

REVIEW_SYSTEM = ("You are a senior embedded engineer reviewing a register map that a junior colleague extracted "
                 "from a datasheet, before it is used in production firmware. You are precise and only change "
                 "what the datasheet text supports. " + JSON_ONLY)

REVIEW_USER = """An automatic checker found these problems in the extracted register map:
{issues}

Extracted register map:
{registers}

Datasheet text (register pages):
{text}

Fix each problem using the datasheet text, keep everything else unchanged, and return the complete corrected
register map in the same JSON format."""

BRIEF_SYSTEM = ("You are a senior hardware engineer writing a one-page brief for a colleague who will wire up this "
                "part and write its driver tomorrow. Be concrete, use the numbers from the data, and warn about "
                "anything that can destroy the part.")

BRIEF_USER = """Write the brief in Markdown with these sections:
## What it is
## Key limits (table: parameter, min, typ, max, unit)
## Wiring checklist (checkboxes)
## Driver start-up sequence (numbered, with register names and values)
## Gotchas

Device: {device}
Limits: {limits}
Registers: {registers}
Scale factors: {scales}"""

ASK_SYSTEM = ("You answer questions about one datasheet using only the excerpts given. Cite the page of every fact "
              "as [p. N]. If the excerpts do not contain the answer, say exactly: \"Not found in this datasheet.\"")

ASK_USER = """Question: {question}

Excerpts:
{excerpts}"""

# ------------------------------------------------------------------ quality notes

CLASSIFY_NOTES_SYSTEM = ("You label free-text failure notes written by operators at end-of-line test stations "
                         "in a bearing and actuator factory. " + JSON_ONLY)

CLASSIFY_NOTES_USER = """Allowed failure modes: {modes}.
severity: minor (cosmetic, part can be reworked), major (part scrapped), critical (safety risk or many parts affected).
test_equipment_suspected: true if the note suggests the tester, fixture or gauge may be wrong rather than the part.

Examples:
note 901: "noise test fail, rumble at high rpm, bearing 6204, retest also fail" ->
{{"note_id": 901, "failure_mode": "noise_vibration", "component": "bearing", "severity": "major", "test_equipment_suspected": false}}
note 902: "bore dia 20.012 over tol (max 20.008), whole tray from grinder 3 check!!" ->
{{"note_id": 902, "failure_mode": "dimension_out_of_tolerance", "component": "inner ring", "severity": "critical", "test_equipment_suspected": false}}
note 903: "torque high on fixture B again, same part passed on fixture A" ->
{{"note_id": 903, "failure_mode": "test_equipment", "component": "test fixture", "severity": "minor", "test_equipment_suspected": true}}

Label these notes. Return {{"labels": [ ... one object per note, in the same format ... ]}}.
{notes}"""

NAME_CLUSTERS_SYSTEM = "You name groups of similar factory failure notes. " + JSON_ONLY

NAME_CLUSTERS_USER = """Each group below was formed automatically from similar wording.
Give each group a short name (max 5 words) and a one-sentence description of what the notes have in common.
Return {{"clusters": [{{"cluster": <number>, "name": "...", "description": "..."}}, ...]}}.

{groups}"""

ROOT_CAUSE_SYSTEM = ("You are an experienced quality engineer leading an 8D problem-solving team in a bearing factory. "
                     "You reason from the evidence in the data and say how confident you are.")

ROOT_CAUSE_USER = """Failure mode: {mode}
Count: {count} of {total} failures

Statistics:
{stats}

Operator notes:
{notes}

Think step by step: (1) what do the notes have in common (station, product, shift, wording)?
(2) which causes would explain that pattern? (3) what evidence supports or weakens each one?
(4) what should be checked first, and what containment is needed today?

Then give the final result between <answer> and </answer> as
{{"failure_mode": "{mode}", "likely_causes": [{{"cause": "...", "evidence": "...", "confidence": "low|medium|high"}}],
"checks": ["..."], "containment": "..."}}"""

SHIFT_REPORT_SYSTEM = ("You are the quality engineer presenting at the 6 a.m. shift meeting. You write short, clear "
                       "Markdown for production supervisors: numbers first, then actions with owners.")

SHIFT_REPORT_USER = """Write the shift-meeting quality report (max 350 words) with:
## Summary (3 bullets with numbers)
## Top problems (table: failure mode, count, share, trend)
## Root-cause status
## Actions for today (checkboxes, each with an owner role)

Period: {period}
Totals: {totals}
Pareto: {pareto}
Root-cause analyses: {root_causes}"""
