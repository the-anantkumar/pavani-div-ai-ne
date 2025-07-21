import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import AstroPersonalityBot


@pytest.fixture
def bot():
    return AstroPersonalityBot


def test_generate_random_birth_data(bot):
    b = bot()
    data = b.generate_random_birth_data()
    assert {"date", "time", "lat", "lon", "city", "country"}.issubset(data.keys())

    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "locations.csv")
    row = df[(df["city"] == data["city"]) & (df["country"] == data["country"])]
    assert not row.empty
    assert float(row.iloc[0]["latitude"]) == pytest.approx(data["lat"])
    assert float(row.iloc[0]["longitude"]) == pytest.approx(data["lon"])


def test_generate_chart_json_structure(bot):
    try:
        import immanuel  # noqa: F401
    except Exception:
        pytest.skip("immanuel not installed")
    b = bot()
    birth = b.generate_random_birth_data()
    chart_json = b.generate_chart_json(birth)
    obj = json.loads(chart_json)
    assert "planets" in obj and "angles" in obj
