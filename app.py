"""Streamlined Astrological Personality Chatbot.

This application generates an astrological chart using the `immanuel` library,
feeds the JSON data to a language model for interpretation and provides a chat
interface via Gradio or FastAPI.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:  # Optional imports for test environments
    import gradio as gr
except Exception:  # pragma: no cover - optional dependency
    gr = None
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover - optional dependency
    FastAPI = HTTPException = BaseModel = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None
try:
    from immanuel import charts
    from immanuel.const import chart, calc
    from immanuel.setup import settings
except Exception:  # pragma: no cover - optional dependency
    charts = chart = calc = settings = None

# Load available locations once for generating real birth places
LOCATIONS_DF = pd.read_csv(Path(__file__).resolve().parent / "data" / "locations.csv")



class AstroPersonalityBot:
    """Main chatbot class handling chart generation and conversations."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self._setup_immanuel()
        self.model_name = model_name
        self.device = "cuda" if torch and hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        try:
            self._setup_llm()
        except Exception as exc:  # pragma: no cover - optional during tests
            print(f"LLM setup skipped: {exc}")
            self.tokenizer = None
            self.model = None
        self.current_chart_json: Optional[str] = None
        self.current_personality: Optional[Dict[str, Any]] = None
        self.chat_history: List[Tuple[str, str]] = []

    def _setup_immanuel(self) -> None:
        if not settings:
            return
        settings.house_system = chart.PLACIDUS
        settings.objects = [
            chart.SUN,
            chart.MOON,
            chart.MERCURY,
            chart.VENUS,
            chart.MARS,
            chart.JUPITER,
            chart.SATURN,
            chart.URANUS,
            chart.NEPTUNE,
            chart.PLUTO,
            chart.ASC,
            chart.MC,
            chart.DESC,
            chart.IC,
        ]
        settings.aspects = [
            calc.CONJUNCTION,
            calc.OPPOSITION,
            calc.SQUARE,
            calc.TRINE,
            calc.SEXTILE,
        ]
        settings.default_orb = 8.0

    def _setup_llm(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers not available")
        if torch is None:
            raise ImportError("torch not available")
        print(f"Loading {self.model_name}...")
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        print("Model loaded successfully!")

    def generate_random_birth_data(self) -> Dict[str, Any]:
        start_date = datetime(1950, 1, 1)
        end_date = datetime(2005, 12, 31)
        random_days = random.randint(0, (end_date - start_date).days)
        birth_date = start_date + timedelta(days=random_days)
        birth_datetime = (
            datetime.combine(birth_date.date(), datetime.min.time())
            + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        )

        location = LOCATIONS_DF.sample(1).iloc[0]
        lat = float(location["latitude"])
        lon = float(location["longitude"])

        return {
            "date": birth_datetime.strftime("%Y-%m-%d"),
            "time": birth_datetime.strftime("%H:%M"),
            "lat": lat,
            "lon": lon,
            "city": location["city"],
            "country": location["country"],
        }

    def generate_chart_json(self, birth_data: Dict[str, Any]) -> str:
        """Convert ``birth_data`` into an ``immanuel`` ``Subject`` and return
        a JSON representation of the generated natal chart."""

        subject = charts.Subject(
            date_time=f"{birth_data['date']} {birth_data['time']}",
            latitude=birth_data['lat'],
            longitude=birth_data['lon'],
            timezone="UTC",
        )
        natal = charts.Natal(subject)

        chart_dict: Dict[str, Any] = {
            "birth_data": birth_data,
            "planets": {},
            "houses": {},
            "aspects": [],
            "angles": {},
        }

        for obj in natal.objects.values():
            if obj.name in {"Asc", "MC", "Desc", "IC"}:
                chart_dict["angles"][obj.name] = {
                    "sign": obj.sign.name,
                    "degree": float(obj.longitude.raw),
                }
            elif obj.type == "Planet":
                chart_dict["planets"][obj.name] = {
                    "sign": obj.sign.name,
                    "degree": float(obj.longitude.raw),
                    "house": obj.house.number if obj.house else None,
                    "retrograde": getattr(obj, "retrograde", False),
                }

        for house in natal.houses.values():
            chart_dict["houses"][f"House_{house.number}"] = {
                "sign": house.sign.name,
                "degree": float(house.longitude.raw),
            }

        for aspect_list in natal.aspects.values():
            for aspect in aspect_list.values():
                chart_dict["aspects"].append(
                    {
                        "planet1": aspect._active_name,
                        "planet2": aspect._passive_name,
                        "type": aspect.type,
                        "angle": float(aspect.difference.degrees),
                        "orb": float(aspect.orb),
                    }
                )
        return json.dumps(chart_dict, indent=2)

    def generate_new_personality(self) -> Tuple[str, str, Dict[str, Any]]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "Language model not loaded. Install required packages and re-run."
            )
        birth_data = self.generate_random_birth_data()
        self.current_chart_json = self.generate_chart_json(birth_data)
        chart_data = json.loads(self.current_chart_json)
        print(chart_data)
        return chart_data

if __name__ == "__main__":
    bot = AstroPersonalityBot()
    print(f"setup bot...done!")
