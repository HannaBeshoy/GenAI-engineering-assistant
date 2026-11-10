"""Tool 1: datasheet PDF -> checked register map, limits, scale factors, C header and brief.

Steps (each one model call or one piece of code):
  1 classify_pages     zero-shot          which pages hold registers / limits
  2 extract_registers  few-shot           register map as JSON (chunked for long datasheets)
  3 extract_limits     few-shot           electrical limits as JSON
  4 check (code)                          duplicate addresses, overlapping bits, min<=typ<=max ...
  5 review_registers   persona + findings the model fixes what the checks found, then re-check
  6 derive_scales      chain-of-thought   full-scale settings and sensitivities
  7 C header (code)                       #defines from the checked JSON
  8 engineering_brief  persona, temp 0.7  one-page brief for the engineer
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import prompts as P
from .checks import check_limits, check_registers
from .cheader import make_header
from .errors import BadModelOutput
from .llm import LLMClient, Params
from .retrieval import top_pages
from .schemas import Limits, PageLabels, RegisterMap, ScaleFactors
from .structured import ask_structured

CHUNK_CHARS = 12000          # about 3000 tokens of datasheet text per extraction call


@dataclass
class DatasheetSettings:
    extract: Params = field(default_factory=lambda: Params(temperature=0.0, max_tokens=4096))
    write: Params = field(default_factory=lambda: Params(temperature=0.7, max_tokens=1200))
    answer: Params = field(default_factory=lambda: Params(temperature=0.2, max_tokens=600))
    review_rounds: int = 1


@dataclass
class DatasheetResult:
    page_labels: dict = field(default_factory=dict)
    registers: Optional[RegisterMap] = None
    limits: Optional[Limits] = None
    scales: Optional[ScaleFactors] = None
    issues_found: list = field(default_factory=list)     # before the review
    issues_left: list = field(default_factory=list)      # after the review
    header: str = ""
    brief: str = ""
    warnings: list = field(default_factory=list)


def _pages_text(pages: list, numbers) -> str:
    return "\n\n".join(f"--- page {p.number} ---\n{p.text}" for p in pages if p.number in numbers)


def _chunks(text: str, size: int = CHUNK_CHARS) -> list:
    parts, cur = [], ""
    for block in text.split("\n\n--- page "):
        block = block if block.startswith("--- page ") else "--- page " + block
        if cur and len(cur) + len(block) > size:
            parts.append(cur)
            cur = ""
        cur += ("\n\n" if cur else "") + block
    return [p for p in parts + [cur] if p.strip()]


def run(pages: list, client: LLMClient, settings: DatasheetSettings,
        progress: Callable[[str, float], None] = lambda s, f: None) -> DatasheetResult:
    res = DatasheetResult()
    s = settings

    # 1 classify pages ---------------------------------------------------------------
    progress("Classifying pages (zero-shot)", 0.05)
    listing = "\n\n".join(f"[page {p.number}]\n{p.text[:700]}" for p in pages)
    labels = ask_structured(client, task="classify_pages", technique=P.TECHNIQUES["classify_pages"],
                            system=P.CLASSIFY_PAGES_SYSTEM, user=P.CLASSIFY_PAGES_USER.format(pages=listing),
                            schema=PageLabels, params=s.extract)
    res.page_labels = {pl.page: pl.category for pl in labels.pages if 1 <= pl.page <= len(pages)}
    reg_pages = [n for n, c in res.page_labels.items() if c in ("register_map", "register_detail")]
    lim_pages = [n for n, c in res.page_labels.items() if c == "electrical"]
    if not reg_pages:
        res.warnings.append("No register pages were found; using every page for register extraction.")
        reg_pages = [p.number for p in pages]
    if not lim_pages:
        res.warnings.append("No electrical-characteristics page was found; using every page for limits.")
        lim_pages = [p.number for p in pages]
    reg_text = _pages_text(pages, reg_pages)

    # 2 registers (chunked) ------------------------------------------------------------
    merged, device = {}, ""
    chunks = _chunks(reg_text)
    for i, chunk in enumerate(chunks, 1):
        progress(f"Extracting registers (few-shot), part {i}/{len(chunks)}", 0.1 + 0.3 * i / len(chunks))
        part = ask_structured(client, task="extract_registers", technique=P.TECHNIQUES["extract_registers"],
                              system=P.EXTRACT_REGISTERS_SYSTEM, user=P.EXTRACT_REGISTERS_USER.format(text=chunk),
                              schema=RegisterMap, params=s.extract)
        device = device or part.device
        for r in part.registers:
            key = r.address.lower()
            if key not in merged or len(r.fields) > len(merged[key].fields):
                merged[key] = r
            elif merged[key].name != r.name:
                merged[f"{key}#{r.name}"] = r      # keep conflicts so the checks report them
    rmap = RegisterMap(device=device or "DEVICE", registers=list(merged.values()))

    # 3 limits -------------------------------------------------------------------------
    progress("Extracting electrical limits (few-shot)", 0.45)
    res.limits = ask_structured(client, task="extract_limits", technique=P.TECHNIQUES["extract_limits"],
                                system=P.EXTRACT_REGISTERS_SYSTEM.replace("register maps", "electrical limits"),
                                user=P.EXTRACT_LIMITS_USER.format(text=_pages_text(pages, lim_pages)[:CHUNK_CHARS]),
                                schema=Limits, params=s.extract)

    # 4-5 check, then review -------------------------------------------------------------
    progress("Checking the extracted data", 0.55)
    res.issues_found = check_registers(rmap) + check_limits(res.limits)
    reg_issues = check_registers(rmap)
    for _ in range(s.review_rounds):
        if not reg_issues:
            break
        progress(f"Senior-engineer review of {len(reg_issues)} problem(s) (persona)", 0.6)
        try:
            rmap = ask_structured(
                client, task="review_registers", technique=P.TECHNIQUES["review_registers"],
                system=P.REVIEW_SYSTEM,
                user=P.REVIEW_USER.format(issues="\n".join(f"- {x}" for x in reg_issues),
                                          registers=rmap.model_dump_json(), text=reg_text[:CHUNK_CHARS]),
                schema=RegisterMap, params=s.extract)
        except BadModelOutput as e:
            res.warnings.append(f"Review step failed, keeping the unreviewed map: {e}")
            break
        reg_issues = check_registers(rmap)
    res.registers = rmap
    res.issues_left = reg_issues + check_limits(res.limits)

    # 6 scale factors --------------------------------------------------------------------
    progress("Deriving scale factors (chain-of-thought)", 0.7)
    try:
        res.scales = ask_structured(client, task="derive_scales", technique=P.TECHNIQUES["derive_scales"],
                                    system=P.DERIVE_SCALES_SYSTEM,
                                    user=P.DERIVE_SCALES_USER.format(text=reg_text[:CHUNK_CHARS]),
                                    schema=ScaleFactors, params=s.extract)
    except BadModelOutput as e:
        res.warnings.append(f"No scale factors: {e}")

    # 7 header -------------------------------------------------------------------------
    progress("Writing the C header", 0.8)
    res.header = make_header(rmap, res.scales)

    # 8 brief --------------------------------------------------------------------------
    progress("Writing the engineering brief (persona)", 0.85)
    res.brief = client.complete(
        "engineering_brief", P.TECHNIQUES["engineering_brief"], P.BRIEF_SYSTEM,
        P.BRIEF_USER.format(device=rmap.device, limits=res.limits.model_dump_json(),
                            registers=json.dumps([r.model_dump() for r in rmap.registers])[:8000],
                            scales=res.scales.model_dump_json() if res.scales else "none"),
        s.write)
    progress("Done", 1.0)
    return res


def ask(pages: list, question: str, client: LLMClient, params: Params) -> tuple:
    """Answer one question from the most relevant pages; returns (answer, page numbers used)."""
    question = question.strip()
    if not question:
        return "Type a question first.", []
    hits = top_pages(pages, question)
    if not hits:
        return "Not found in this datasheet.", []
    excerpts = "\n\n".join(f"[p. {n}]\n{text}" for n, text, _ in hits)
    answer = client.complete("ask_datasheet", P.TECHNIQUES["ask_datasheet"], P.ASK_SYSTEM,
                             P.ASK_USER.format(question=question, excerpts=excerpts), params)
    return answer, [n for n, _, _ in hits]


def as_json(res: DatasheetResult) -> str:
    return json.dumps({
        "device": res.registers.device if res.registers else None,
        "registers": [r.model_dump() for r in res.registers.registers] if res.registers else [],
        "limits": [lim.model_dump() for lim in res.limits.limits] if res.limits else [],
        "scales": [x.model_dump() for x in res.scales.scales] if res.scales else [],
        "issues_left": res.issues_left,
    }, indent=2, ensure_ascii=False)
