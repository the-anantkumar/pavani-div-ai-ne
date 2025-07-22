"""Streamlined Astrological Personality Chatbot.

This application generates an astrological chart using the `immanuel` library,
feeds the JSON data to a language model for interpretation and provides a chat
interface via Gradio or FastAPI.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from immanuel import charts # type: ignore
from immanuel.const import chart, calc # type: ignore
from immanuel.setup import settings # type: ignore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class AstroPersonalityBot:
    """Main chatbot class handling chart generation and conversations."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        logger.info("Initializing AstroPersonalityBot...")
        self._setup_immanuel()
        logger.info("Immanuel setup complete")
        
        self.model_name = model_name
        # Check for CUDA availability once
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info("Starting LLM setup...")
            self._setup_llm()
            logger.info("LLM setup complete")
        except ImportError as exc:
            logger.error(f"LLM setup skipped: Required module not available - {exc}")
            self.tokenizer = None
            self.model = None
        except RuntimeError as exc:
            logger.error(f"LLM setup skipped: Runtime error - {exc}")
            self.tokenizer = None
            self.model = None
        except Exception as exc:
            logger.error(f"LLM setup skipped: Unexpected error - {exc.__class__.__name__}: {exc}")
            self.tokenizer = None
            self.model = None
        
        self.current_chart_json: Optional[str] = None
        self.current_personality: Optional[Dict[str, Any]] = None
        self.chat_history: List[Tuple[str, str]] = []
        logger.info("AstroPersonalityBot initialization complete")

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
        
        logger.info(f"Loading model: {self.model_name}")
        
        if self.device == "cuda":
            logger.info("Setting up 4-bit quantization config for CUDA")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            logger.info("No quantization config for CPU")
            bnb_config = None

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
        
        logger.info("Loading model (this may take several minutes and use significant memory)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        logger.info("Model loaded successfully!")

    def generate_random_birth_data(self) -> Dict[str, Any]:
        # Generate a random date between 1950 and 2005
        start_date = datetime(1950, 1, 1)
        end_date = datetime(2005, 12, 31)
        random_days = random.randint(0, (end_date - start_date).days)
        birth_date = start_date + timedelta(days=random_days)
        
        # Add random hour and minute
        random_hour = random.randint(0, 23)
        random_minute = random.randint(0, 59)
        birth_datetime = datetime(
            birth_date.year, birth_date.month, birth_date.day,
            random_hour, random_minute
        )

        # Generate random coordinates (avoid polar regions for house calculations)
        lat = random.uniform(-60, 60)  # Avoid polar circles
        lon = random.uniform(-180, 180)

        return {
            "date": birth_datetime.strftime("%Y-%m-%d"),
            "time": birth_datetime.strftime("%H:%M"),
            "lat": lat,
            "lon": lon,
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

        # Process aspects more efficiently with a list comprehension
        chart_dict["aspects"] = [
            {
                "planet1": aspect._active_name,
                "planet2": aspect._passive_name,
                "type": aspect.type,
                "angle": float(aspect.difference.degrees),
                "orb": float(aspect.orb),
            }
            for aspect_list in natal.aspects.values()
            for aspect in aspect_list.values()
        ]
        return json.dumps(chart_dict, indent=2)

    def interpret_chart_with_llm(self, chart_json: str) -> Dict[str, Any]:
        """Send birth chart JSON to LLM for personality interpretation."""
        logger.info("Starting LLM interpretation...")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Language model not loaded")
        
        logger.info("Creating prompt...")
        prompt = f"""<s>[INST] You are an expert astrologer. Analyze this birth chart and provide a personality interpretation focusing on key traits, strengths, and characteristics. Be specific and insightful.

                    Birth Chart Data:
                    {chart_json}

                    Provide a structured personality analysis covering:
                    1. Core personality traits
                    2. Strengths and talents
                    3. Potential challenges
                    4. Communication style
                    5. Relationship patterns

                    Keep the response concise but meaningful. [/INST]"""
        
        logger.info("Tokenizing input...")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logger.info("Generating response with model...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        logger.info("Decoding response...")
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part after the prompt
        if "[/INST]" in response:
            interpretation = response.split("[/INST]")[-1].strip()
        else:
            # Fallback: remove the input prompt from response
            interpretation = response[len(prompt):].strip()
        logger.info("LLM interpretation completed successfully")
        
        return {
            "interpretation": interpretation,
            "chart_data": json.loads(chart_json)
        }

    def generate_birth_chart(self) -> Dict[str, Any]:
        logger.info("Starting birth chart generation...")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "Language model not loaded. Install required packages and re-run."
            )
        
        logger.info("Generating random birth data...")
        birth_data = self.generate_random_birth_data()
        
        logger.info("Creating chart JSON...")
        self.current_chart_json = self.generate_chart_json(birth_data)
        
        try:
            chart_data = json.loads(self.current_chart_json)
            logger.info("Chart JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing chart JSON: {e}")
            chart_data = {"error": "Failed to parse chart data", "raw": self.current_chart_json[:100] + '...'}
            self.current_chart_json = json.dumps(chart_data)
        
        logger.info("Starting LLM interpretation...")
        self.current_personality = self.interpret_chart_with_llm(self.current_chart_json)
        logger.info("LLM interpretation complete")
        
        print("Birth Chart Generated:")
        print(json.dumps(chart_data, indent=2))
        print("\nPersonality Interpretation:")
        print(self.current_personality["interpretation"])
        
        return chart_data

    def get_personality(self) -> Optional[str]:
        """Get the current personality interpretation."""
        if self.current_personality:
            return self.current_personality["interpretation"]
        return None

if __name__ == "__main__":
    bot = AstroPersonalityBot()
    print(f"setup bot...done!")
