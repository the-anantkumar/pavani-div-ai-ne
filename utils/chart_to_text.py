"""Utility functions to convert chart data to readable text."""
from __future__ import annotations
from typing import Dict, Any


def chart_to_summary(chart_data: Dict[str, Any]) -> str:
    """Return a short summary of key placements from parsed chart data."""
    planets = chart_data.get("planets", {})
    angles = chart_data.get("angles", {})
    if not planets or not angles:
        return "No chart information available."

    sun = planets.get("Sun", {}).get("sign")
    moon = planets.get("Moon", {}).get("sign")
    asc = angles.get("Asc", {}).get("sign")
    return (
        f"Sun in {sun}, Moon in {moon}, Ascendant in {asc}."
        if sun and moon and asc
        else "Chart data incomplete."
    )
