"""Summary construction for the independent phase 6A.1 workflow."""

from __future__ import annotations

from grindcae import __version__

from .uniaxial import UniaxialHistoryResult


RESULT_FORMAT = "grindcae_phase_6a1_j2_material_point"


def build_summary(
    result: UniaxialHistoryResult, artifact_paths: dict[str, str]
) -> dict[str, object]:
    """Build the deterministic JSON-serializable phase 6A.1 summary."""

    if not isinstance(result, UniaxialHistoryResult):
        raise TypeError("result must be an UniaxialHistoryResult")
    material = result.case.material
    first_yield = (
        None
        if result.first_yield_step_id is None
        else result.steps[result.first_yield_step_id].to_dict()
    )
    return {
        "result_format": RESULT_FORMAT,
        "package_version": __version__,
        "material_point_schema_version": result.case.material_point_schema_version,
        "unit_system": result.case.unit_system,
        "model_type": result.case.model_type,
        "normalized_input": result.case.to_dict(),
        "units": {
            "stress": "Pa",
            "strain": "dimensionless",
            "elastic_modulus": "Pa",
            "tangent_modulus": "Pa",
            "internal_hardening_modulus": "Pa",
        },
        "material": {
            **material.to_dict(),
            "internal_hardening_modulus_Pa": material.internal_hardening_modulus,
        },
        "uniaxial_response": {
            "yield_strain": result.yield_strain,
            "first_yield_step": first_yield,
            "peak": result.peak_step.to_dict(),
            "final_unloaded": result.final_step.to_dict(),
            "maximum_absolute_transverse_stress_Pa": (
                result.maximum_absolute_transverse_stress_Pa
            ),
            "maximum_plastic_yield_function_residual_Pa": (
                result.maximum_plastic_yield_function_residual_Pa
            ),
            "step_counts": {
                "loading": result.case.history.loading_step_count,
                "unloading": result.case.history.unloading_step_count,
                "total_converged_states_including_initial": len(result.steps),
            },
        },
        "assumptions_and_limitations": [
            "This is an independent small-strain material point with no mesh or spatial field.",
            "The constitutive model is three-dimensional J2/von Mises associative plasticity with bilinear isotropic hardening.",
            "The benchmark applies x-direction uniaxial stress; both lateral stresses and all shear stresses are controlled to zero.",
            "The history contains monotonic tensile loading followed by elastic unloading to zero axial stress.",
            "There is no reverse compression, no cyclic plasticity, no kinematic hardening, and no Bauschinger effect.",
            "There is no FEM, no grinding geometry or load, no 4C4 history transfer, no GUI, and no portable-package integration.",
            "The result does not model material removal, damage, contact, temperature, roughness, or industrial validation.",
        ],
        "artifacts": {
            "summary_json": {
                "filename": "summary.json",
                "path": artifact_paths["summary_json"],
                "definition": "Normalized input, material parameters, benchmark response, assumptions, and artifact traceability.",
            },
            "history_csv": {
                "filename": "history.csv",
                "path": artifact_paths["history_csv"],
                "definition": "One 0-based row for each converged initial, loading, and unloading material-point state.",
            },
            "stress_strain_png": {
                "filename": "stress_strain.png",
                "path": artifact_paths["stress_strain_png"],
                "definition": "Axial total-strain versus axial-stress material-point curve with yield, peak, and residual-strain markers.",
            },
        },
    }
