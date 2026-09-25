"""Versioned provisional and secondary abrasive-grit references."""

from __future__ import annotations


TABLE_VERSION = "phase_7a2_provisional_table_v1"
W_SECONDARY_TABLE_VERSION = "phase_7a2_1_w_secondary_reference_table_v1"
FEPA_PUBLIC_TABLE_VERSION = "washington_mills_fepa_public_conversion_2026_08_27_v1"
JIS_SECONDARY_TABLE_VERSION = "phase_7a2_2_jis_secondary_reference_table_v1"

FEPA_PUBLIC_DATA_STATUS = (
    "public_secondary_reference_unverified_against_primary_standard"
)
FEPA_PUBLIC_STANDARD_STATUS = (
    "washington_mills_page_claims_fepa_42_1_2006_not_verified_against_primary_standard"
)
FEPA_PUBLIC_SOURCE = (
    "Washington Mills, FEPA Particle Size Conversion Chart, "
    "https://www.washingtonmills.com/resources/guides/fepa-particle-size-conversion-chart "
    "(accessed 2026-08-27). The page states that macrogrit micron size is based on the "
    "40% minimum sieve and that F grits conform to FEPA 42-1:2006; the primary standard "
    "and a product inspection report were not independently verified."
)

W_SECONDARY_DATA_STATUS = "secondary_reference_unverified"
W_SECONDARY_STANDARD_STATUS = (
    "claimed_chinese_w_micropowder_reference_not_verified_against_primary_standard"
)
W_SECONDARY_SOURCE = (
    "User-provided screenshot of a secondary abrasive grit comparison table. "
    "The original standard number, edition, publisher, and primary source have not been verified."
)
JIS_SECONDARY_DATA_STATUS = "secondary_reference_unverified"
JIS_SECONDARY_STANDARD_STATUS = (
    "japanese_hash_label_from_unverified_secondary_comparison_not_primary_JIS"
)
JIS_SECONDARY_SOURCE = (
    W_SECONDARY_SOURCE
    + " The Japanese hash label and the paired W-series diameter interval are retained "
    "only as an explicit secondary comparison; this does not establish official JIS equivalence."
)

GRIT_TABLES: dict[str, dict[str, dict[str, object]]] = {
    "GB_FEPA_F": {
        **{
            designation: {
                "diameter_lower_m": diameter_m,
                "representative_diameter_d50_m": diameter_m,
                "diameter_upper_m": diameter_m,
                "representative_diameter_basis": "washington_mills_public_40_percent_minimum_sieve_reference_not_measured_d50",
                "range_definition": "single_public_40_percent_minimum_sieve_reference_no_distribution_available",
                "standard_status": FEPA_PUBLIC_STANDARD_STATUS,
                "data_status": FEPA_PUBLIC_DATA_STATUS,
                "source": FEPA_PUBLIC_SOURCE,
                "table_version": FEPA_PUBLIC_TABLE_VERSION,
            }
            for designation, diameter_m in (
                ("F4", 4750e-6),
                ("F5", 4000e-6),
                ("F6", 3350e-6),
                ("F7", 2800e-6),
                ("F8", 2360e-6),
                ("F10", 2000e-6),
                ("F12", 1700e-6),
                ("F14", 1400e-6),
                ("F16", 1180e-6),
                ("F20", 1000e-6),
                ("F22", 850e-6),
                ("F24", 710e-6),
                ("F30", 600e-6),
                ("F36", 500e-6),
                ("F40", 425e-6),
                ("F46", 355e-6),
                ("F54", 300e-6),
                ("F60", 250e-6),
                ("F70", 212e-6),
                ("F90", 150e-6),
                ("F100", 125e-6),
                ("F120", 106e-6),
                ("F150", 75e-6),
                ("F180", 63e-6),
                ("F220", 53e-6),
            )
        },
        "F80": {
            "diameter_lower_m": 150e-6,
            "representative_diameter_d50_m": 185e-6,
            "diameter_upper_m": 212e-6,
            "range_definition": "provisional_lower_d50_upper",
            "data_status": "provisional_reference_pending_standard_verification",
            "source": "Phase 7A.2 provisional GB/FEPA F engineering reference; verify against the applicable standard edition before engineering use.",
        }
    },
    "JIS_hash": {
        **{
            designation: {
                "diameter_lower_m": lower_m,
                "diameter_upper_m": upper_m,
                "representative_diameter_basis": "geometric_mean_of_secondary_w_comparison_range_not_measured_d50",
                "range_definition": "secondary_w_comparison_lower_upper_geometric_mean_representative",
                "standard_status": JIS_SECONDARY_STANDARD_STATUS,
                "data_status": JIS_SECONDARY_DATA_STATUS,
                "source": JIS_SECONDARY_SOURCE,
                "table_version": JIS_SECONDARY_TABLE_VERSION,
            }
            for designation, lower_m, upper_m in (
                ("#450", 30.0e-6, 40.0e-6),
                ("#700", 20.0e-6, 30.0e-6),
                ("#1000", 12.0e-6, 22.0e-6),
                ("#1800", 6.0e-6, 12.0e-6),
                ("#4000", 3.0e-6, 6.0e-6),
                ("#6000", 2.0e-6, 4.0e-6),
            )
        },
        "#5000": {
            "diameter_lower_m": 3.4e-6,
            "representative_diameter_d50_m": 3.4e-6,
            "diameter_upper_m": 3.4e-6,
            "range_definition": "single_nominal_representative_diameter_no_distribution_available",
            "standard_status": "assumed_JIS_for_phase_7a2_demonstration",
            "data_status": "provisional_pending_manufacturer_verification",
            "source": "Phase 7A.2 assumes the user-provided #5000 aluminum-oxide wheel follows a JIS-style designation and uses an approximate representative diameter of 3.4 micrometres. Replace with manufacturer data when available.",
        }
    },
    "GB_W_micropowder": {
        "W3.5": {
            "diameter_lower_m": 2.0e-6,
            "diameter_upper_m": 4.0e-6,
            "related_japanese_hash_designation": "#6000",
        },
        "W5": {
            "diameter_lower_m": 3.0e-6,
            "diameter_upper_m": 6.0e-6,
            "related_japanese_hash_designation": "#4000",
        },
        "W10": {
            "diameter_lower_m": 6.0e-6,
            "diameter_upper_m": 12.0e-6,
            "related_japanese_hash_designation": "#1800",
        },
        "W20": {
            "diameter_lower_m": 12.0e-6,
            "diameter_upper_m": 22.0e-6,
            "related_japanese_hash_designation": "#1000",
        },
        "W28": {
            "diameter_lower_m": 20.0e-6,
            "diameter_upper_m": 30.0e-6,
            "related_japanese_hash_designation": "#700",
        },
        "W40": {
            "diameter_lower_m": 30.0e-6,
            "diameter_upper_m": 40.0e-6,
            "related_japanese_hash_designation": "#450",
        },
        "W50": {
            "diameter_lower_m": 36.0e-6,
            "diameter_upper_m": 54.0e-6,
            "related_japanese_hash_designation": None,
        },
    },
}
