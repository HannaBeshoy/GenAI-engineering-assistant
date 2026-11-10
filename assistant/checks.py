"""Rule-based checks on what the model extracted. The model is good at reading tables but
can still duplicate a register or overlap two bit fields; these checks catch that in code,
and their findings are sent back to the model for a reviewed correction."""
from __future__ import annotations

from .schemas import Limits, RegisterMap

KNOWN_UNITS = {"V", "mV", "A", "mA", "µA", "uA", "nA", "W", "mW", "°C", "C", "K", "Hz", "kHz", "MHz",
               "s", "ms", "µs", "us", "ns", "g", "dps", "°/s", "LSB/g", "LSB/(°/s)", "LSB/°C", "%", "pF",
               "kΩ", "Ω", "ohm", "kohm", "V/ms", "mg", "mg/√Hz", "°/s/√Hz", "bit", "bits"}


def _bit_set(bits: str) -> set:
    if ":" in bits:
        hi, lo = (int(x) for x in bits.split(":"))
        if hi < lo:
            hi, lo = lo, hi
        return set(range(lo, hi + 1))
    return {int(bits)}


def check_registers(rmap: RegisterMap) -> list:
    issues = []
    seen_addr, seen_name = {}, {}
    for r in rmap.registers:
        a = r.address.lower()
        if a in seen_addr:
            issues.append(f"{r.name}: address {r.address} is also used by {seen_addr[a]}")
        seen_addr[a] = r.name
        if r.name in seen_name:
            issues.append(f"{r.name}: register name appears twice")
        seen_name[r.name] = True
        used = set()
        for f in r.fields:
            if ":" in f.bits and int(f.bits.split(":")[0]) < int(f.bits.split(":")[1]):
                issues.append(f"{r.name}.{f.name}: bit range {f.bits} is written low:high, expected high:low")
            b = _bit_set(f.bits)
            if used & b:
                issues.append(f"{r.name}.{f.name}: bits {f.bits} overlap another field of {r.name}")
            used |= b
        if r.access == "R" and r.fields and any("enable" in f.description.lower() for f in r.fields):
            issues.append(f"{r.name}: read-only register has an 'enable' field - check the access type")
    return issues


def check_limits(limits: Limits) -> list:
    issues = []
    for lim in limits.limits:
        vals = [v for v in (lim.min, lim.typ, lim.max) if v is not None]
        if not vals:
            issues.append(f"{lim.parameter}: no min, typ or max value")
        if vals != sorted(vals):
            issues.append(f"{lim.parameter}: values are not in min <= typ <= max order ({lim.min}, {lim.typ}, {lim.max})")
        if lim.unit not in KNOWN_UNITS:
            issues.append(f"{lim.parameter}: unusual unit '{lim.unit}'")
    return issues
