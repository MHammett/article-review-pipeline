"""EPA (Environmental Protection Agency) resolver.

EPA does not expose one unified claims-queryable API the way FRED or EIA do —
Envirofacts and ECHO are both record-lookup APIs keyed on facility ID, permit
number, or geography, not free-text claims. Rather than guess a mismatched
facility for an arbitrary claim, this adapter takes the FHWA pointer-only
approach: keyword-gate on the EPA program a claim is actually about and point
at that program's public data portal for manual retrieval.
"""

import logging

from ..topic_match import topic_match

log = logging.getLogger(__name__)

# Ordered so the first (most specific) category to match wins — e.g. a PFAS
# claim should point at the PFAS portal, not the generic water program page.
_CATEGORIES = [
    (
        "PFAS",
        ["pfas", "forever chemical", "perfluoroalkyl", "polyfluoroalkyl"],
        "https://www.epa.gov/pfas",
    ),
    (
        "Greenhouse Gas Reporting Program (GHGRP/FLIGHT)",
        [
            "greenhouse gas",
            "ghg emissions",
            "ghgrp",
            "co2 emissions",
            "carbon emissions",
            "methane emissions",
        ],
        "https://ghgdata.epa.gov/ghgp/main.do",
    ),
    (
        "Toxics Release Inventory (TRI)",
        ["toxic release inventory", "tri program", "toxic release"],
        "https://www.epa.gov/toxics-release-inventory-tri-program",
    ),
    (
        "Air Quality System (AirData / NAAQS)",
        [
            "air quality",
            "naaqs",
            "pm2.5",
            "pm 2.5",
            "particulate matter",
            "ozone",
            "clean air act",
        ],
        "https://www.epa.gov/outdoor-air-quality-data",
    ),
    (
        "ECHO / NPDES Water Program",
        [
            "npdes",
            "clean water act",
            "wastewater discharge",
            "effluent",
            "water permit",
        ],
        "https://echo.epa.gov/",
    ),
    (
        "ECHO Enforcement and Compliance History",
        ["enforcement action", "compliance history", "epa violation", "consent decree"],
        "https://echo.epa.gov/",
    ),
]

_GENERIC_TERMS = [
    "environmental protection agency",
    " epa ",
    "epa regulation",
    "epa data",
]


def resolve(claim, api_key=None):
    """
    Points at the relevant EPA data portal for manual retrieval. Not a
    verified structured-data fetch — see the module docstring for why.
    """
    claim_lower = f" {claim.lower()} "

    for program, terms, url in _CATEGORIES:
        if topic_match(claim_lower, terms):
            return {
                "found": True,
                "pointer_only": True,
                "url": url,
                "summary": f"EPA {program} — manual retrieval required. See {url}",
                "content": f"EPA source pointer ({program}) for: {claim[:200]}",
            }

    if topic_match(claim_lower, _GENERIC_TERMS):
        url = "https://enviro.epa.gov/"
        return {
            "found": True,
            "pointer_only": True,
            "url": url,
            "summary": f"EPA Envirofacts (general) — manual retrieval required. See {url}",
            "content": f"EPA source pointer (Envirofacts) for: {claim[:200]}",
        }

    return {"found": False}
