"""The typed results each step must produce. pydantic checks every model answer against these."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------- datasheet

PAGE_CATEGORIES = ("overview", "pinout", "electrical", "timing", "register_map", "register_detail", "other")


class PageLabel(BaseModel):
    page: int = Field(ge=1)
    category: Literal["overview", "pinout", "electrical", "timing", "register_map", "register_detail", "other"]


class PageLabels(BaseModel):
    pages: List[PageLabel]


class BitField(BaseModel):
    name: str = Field(min_length=1)
    bits: str = Field(pattern=r"^[0-7](:[0-7])?$", description="'7:5' for bits 7 down to 5, or '3' for one bit")
    description: str = ""


class Register(BaseModel):
    name: str = Field(min_length=1)
    address: str = Field(pattern=r"^0x[0-9A-Fa-f]{2}$")
    access: Literal["R", "W", "RW"]
    reset: str = Field(pattern=r"^0x[0-9A-Fa-f]{2}$")
    description: str = ""
    fields: List[BitField] = []

    @field_validator("name")
    @classmethod
    def c_identifier(cls, v: str) -> str:
        v = v.strip().upper().replace(" ", "_").replace("-", "_")
        if not v.replace("_", "").isalnum() or v[0].isdigit():
            raise ValueError(f"'{v}' is not usable as a C identifier")
        return v


class RegisterMap(BaseModel):
    device: str = Field(min_length=1)
    registers: List[Register] = Field(min_length=1)


class Limit(BaseModel):
    parameter: str
    symbol: Optional[str] = None
    min: Optional[float] = None
    typ: Optional[float] = None
    max: Optional[float] = None
    unit: str
    conditions: Optional[str] = None


class Limits(BaseModel):
    limits: List[Limit] = Field(min_length=1)


class ScaleFactor(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    quantity: Literal["acceleration", "angular_rate", "temperature", "other"]
    reg: str = Field(alias="register")
    field: str
    code: int = Field(ge=0, le=7)
    range: str
    sensitivity: float = Field(gt=0, description="LSB per unit")
    unit: str


class ScaleFactors(BaseModel):
    scales: List[ScaleFactor] = Field(min_length=1)


# ---------------------------------------------------------------- quality notes

FAILURE_MODES = ("noise_vibration", "dimension_out_of_tolerance", "seal_damage", "lubrication",
                 "contamination", "torque_out_of_spec", "electrical_fault", "test_equipment",
                 "labeling_handling", "other")


class NoteLabel(BaseModel):
    note_id: int
    failure_mode: Literal["noise_vibration", "dimension_out_of_tolerance", "seal_damage", "lubrication",
                          "contamination", "torque_out_of_spec", "electrical_fault", "test_equipment",
                          "labeling_handling", "other"]
    component: str
    severity: Literal["minor", "major", "critical"]
    test_equipment_suspected: bool = False


class NoteLabels(BaseModel):
    labels: List[NoteLabel]


class ClusterName(BaseModel):
    cluster: int
    name: str
    description: str


class ClusterNames(BaseModel):
    clusters: List[ClusterName]


class Cause(BaseModel):
    cause: str
    evidence: str
    confidence: Literal["low", "medium", "high"]


class RootCause(BaseModel):
    failure_mode: str
    likely_causes: List[Cause] = Field(min_length=1)
    checks: List[str] = Field(min_length=1)
    containment: str
