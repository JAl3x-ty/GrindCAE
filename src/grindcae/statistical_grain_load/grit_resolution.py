"""Resolve strict Phase 7A.2 abrasive-grit variants to one result."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .grit_tables import (
    FEPA_PUBLIC_DATA_STATUS,
    FEPA_PUBLIC_SOURCE,
    FEPA_PUBLIC_STANDARD_STATUS,
    FEPA_PUBLIC_TABLE_VERSION,
    GRIT_TABLES,
    JIS_SECONDARY_DATA_STATUS,
    JIS_SECONDARY_SOURCE,
    JIS_SECONDARY_STANDARD_STATUS,
    JIS_SECONDARY_TABLE_VERSION,
    TABLE_VERSION,
    W_SECONDARY_DATA_STATUS,
    W_SECONDARY_SOURCE,
    W_SECONDARY_STANDARD_STATUS,
    W_SECONDARY_TABLE_VERSION,
)
from .models import (
    ManufacturerWheelSpecification,
    StatisticalGrainLoadValidationError,
    TableWheelSpecification,
    WheelSpecification,
)


def available_grit_designations(resolution_path: str) -> tuple[str, ...]:
    """Return the existing designations for one built-in resolution path."""

    if resolution_path == "manufacturer_override":
        return ()
    table = GRIT_TABLES.get(resolution_path)
    if table is None:
        raise ValueError(f"未知粒度路径：{resolution_path}")
    if resolution_path in {"GB_FEPA_F", "JIS_hash"}:
        return tuple(sorted(table, key=lambda designation: int(designation[1:])))
    return tuple(table)


@dataclass(frozen=True, slots=True)
class ResolvedGritSpecification:
    abrasive_material: str
    grit_designation: str
    resolution_path: str
    diameter_lower_m: float
    representative_diameter_d50_m: float
    representative_diameter_basis: str | None
    diameter_upper_m: float
    range_definition: str
    standard_status: str
    data_status: str
    source: str
    table_version: str
    manufacturer: str | None
    product_model: str | None
    manufacturer_override_applied: bool
    related_japanese_hash_designation: str | None
    assumptions_and_cautions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = {
            "abrasive_material": self.abrasive_material,
            "grit_designation": self.grit_designation,
            "resolution_path": self.resolution_path,
            "diameter_lower_m": self.diameter_lower_m,
            "representative_diameter_d50_m": self.representative_diameter_d50_m,
            "diameter_upper_m": self.diameter_upper_m,
            "range_definition": self.range_definition,
            "standard_status": self.standard_status,
            "data_status": self.data_status,
            "source": self.source,
            "table_version": self.table_version,
            "manufacturer": self.manufacturer,
            "product_model": self.product_model,
            "manufacturer_override_applied": self.manufacturer_override_applied,
            "assumptions_and_cautions": list(self.assumptions_and_cautions),
        }
        if self.representative_diameter_basis is not None:
            result["representative_diameter_basis"] = self.representative_diameter_basis
        if self.related_japanese_hash_designation is not None:
            result["related_japanese_hash_designation"] = self.related_japanese_hash_designation
        elif self.resolution_path == "GB_W_micropowder":
            result["related_japanese_hash_designation"] = None
        return result


def resolve_grit_specification(specification: WheelSpecification) -> ResolvedGritSpecification:
    if isinstance(specification, ManufacturerWheelSpecification):
        return ResolvedGritSpecification(
            abrasive_material=specification.abrasive_material,
            grit_designation=specification.grit_designation,
            resolution_path=specification.resolution_path,
            diameter_lower_m=specification.diameter_lower_m,
            representative_diameter_d50_m=specification.representative_diameter_d50_m,
            representative_diameter_basis=None,
            diameter_upper_m=specification.diameter_upper_m,
            range_definition=specification.range_definition,
            standard_status="manufacturer_override_not_a_standard_table_resolution",
            data_status=specification.data_status,
            source=specification.source,
            table_version="manufacturer_override_input_v1",
            manufacturer=specification.manufacturer,
            product_model=specification.product_model,
            manufacturer_override_applied=True,
            related_japanese_hash_designation=None,
            assumptions_and_cautions=(
                "Manufacturer data override the provisional built-in grit table.",
            "The declared source status does not by itself establish experimental accuracy.",
            ),
        )
    if not isinstance(specification, TableWheelSpecification):
        raise StatisticalGrainLoadValidationError("wheel_specification [validated variant]: wrong object type")
    entry = GRIT_TABLES.get(specification.resolution_path, {}).get(specification.grit_designation)
    if entry is None:
        raise StatisticalGrainLoadValidationError(
            f"wheel_specification.grit_designation [known table entry]: unknown grit designation {specification.grit_designation!r} for {specification.resolution_path}; no guessing or extrapolation is allowed"
        )
    is_w_secondary = specification.resolution_path == "GB_W_micropowder"
    is_fepa_public = (
        specification.resolution_path == "GB_FEPA_F"
        and entry.get("data_status") == FEPA_PUBLIC_DATA_STATUS
    )
    is_jis_secondary = (
        specification.resolution_path == "JIS_hash"
        and entry.get("standard_status") == JIS_SECONDARY_STANDARD_STATUS
    )
    lower = float(entry["diameter_lower_m"])
    upper = float(entry["diameter_upper_m"])
    representative = (
        math.sqrt(lower * upper)
        if is_w_secondary or is_jis_secondary
        else float(entry["representative_diameter_d50_m"])
    )
    w_cautions = (
        "This table entry is secondary reference material without a verified standard number and edition.",
        "The stated diameter interval is copied as written and has not been cross-checked against the primary standard.",
        "The representative_diameter_d50_m field is retained for interface compatibility; its value is the geometric mean of the secondary-reference range, not measured d50 or a particle-size inspection result.",
        "Use manufacturer_override to replace this engineering reference before engineering application.",
        "The related Japanese hash designation is copied from an unverified secondary comparison table and does not establish official JIS equivalence.",
    )
    provisional_cautions = (
        "The built-in entry is provisional and replaceable; it is not an official verified JIS value, certified value, or manufacturer-verified value.",
        "Equal lower, d50, and upper values represent one nominal diameter and are not a known physical diameter distribution.",
        "Use manufacturer_override when a product catalogue or inspection report supplies lower, d50, and upper diameters.",
    )
    fepa_public_cautions = (
        "This public conversion-table entry was not checked against the primary FEPA standard edition.",
        "The single micron value is a published 40% minimum-sieve reference, not measured d50 and not a physical particle-size distribution.",
        "Use manufacturer_override when a product catalogue or inspection report supplies the applicable particle-size data.",
    )
    jis_secondary_cautions = (
        "This unverified secondary comparison does not establish official JIS equivalence for the Japanese hash label and W-series diameter interval.",
        "The representative_diameter_d50_m field is the geometric mean of that secondary-reference interval, not measured d50 or a particle-size inspection result.",
        "The primary JIS standard number, edition, and limits were not independently verified.",
        "Use manufacturer_override when a product catalogue or inspection report supplies the applicable particle-size data.",
    )
    standard_status = (
        W_SECONDARY_STANDARD_STATUS
        if is_w_secondary
        else JIS_SECONDARY_STANDARD_STATUS
        if is_jis_secondary
        else FEPA_PUBLIC_STANDARD_STATUS
        if is_fepa_public
        else str(
            entry.get(
                "standard_status",
                "provisional_GB_FEPA_F_engineering_path_pending_standard_verification",
            )
        )
    )
    data_status = (
        W_SECONDARY_DATA_STATUS
        if is_w_secondary
        else JIS_SECONDARY_DATA_STATUS
        if is_jis_secondary
        else FEPA_PUBLIC_DATA_STATUS
        if is_fepa_public
        else str(entry["data_status"])
    )
    source = (
        W_SECONDARY_SOURCE
        if is_w_secondary
        else JIS_SECONDARY_SOURCE
        if is_jis_secondary
        else FEPA_PUBLIC_SOURCE
        if is_fepa_public
        else str(entry["source"])
    )
    table_version = (
        W_SECONDARY_TABLE_VERSION
        if is_w_secondary
        else JIS_SECONDARY_TABLE_VERSION
        if is_jis_secondary
        else FEPA_PUBLIC_TABLE_VERSION
        if is_fepa_public
        else TABLE_VERSION
    )
    return ResolvedGritSpecification(
        abrasive_material=specification.abrasive_material,
        grit_designation=specification.grit_designation,
        resolution_path=specification.resolution_path,
        diameter_lower_m=lower,
        representative_diameter_d50_m=representative,
        representative_diameter_basis=(
            "geometric_mean_of_secondary_reference_range_not_measured_d50"
            if is_w_secondary
            else "geometric_mean_of_secondary_w_comparison_range_not_measured_d50"
            if is_jis_secondary
            else str(entry["representative_diameter_basis"])
            if is_fepa_public
            else None
        ),
        diameter_upper_m=upper,
        range_definition=(
            "secondary_reference_lower_upper_geometric_mean_representative"
            if is_w_secondary
            else "secondary_w_comparison_lower_upper_geometric_mean_representative"
            if is_jis_secondary
            else str(entry["range_definition"])
        ),
        standard_status=standard_status,
        data_status=data_status,
        source=source,
        table_version=table_version,
        manufacturer=None,
        product_model=None,
        manufacturer_override_applied=False,
        related_japanese_hash_designation=(
            entry.get("related_japanese_hash_designation") if is_w_secondary else None
        ),
        assumptions_and_cautions=(
            w_cautions
            if is_w_secondary
            else jis_secondary_cautions
            if is_jis_secondary
            else fepa_public_cautions
            if is_fepa_public
            else provisional_cautions
        ),
    )
