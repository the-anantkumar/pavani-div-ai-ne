"""Streamlined Astrological Personality Chatbot.

This application generates an astrological chart using the `immanuel` library,
feeds the JSON data to a language model for interpretation and provides a chat
interface via Gradio or FastAPI.
"""

from __future__ import annotations

import json
import logging
import random
import signal
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

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        logger.info("Initializing AstroPersonalityBot...")
        self._setup_immanuel()
        
        self.model_name = model_name
        # Detect best available device (MPS for MacBook, CUDA for NVIDIA, CPU fallback)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info("Using device: %s", self.device)
        
        self._initialize_llm()
        self._initialize_state()
        logger.info("AstroPersonalityBot initialization complete")

    def _initialize_llm(self) -> None:
        """Initialize the language model with error handling."""
        try:
            logger.info("Starting LLM setup...")
            self._setup_llm()
            logger.info("LLM setup complete")
        except (ImportError, RuntimeError, Exception) as exc:
            logger.error("LLM setup skipped: %s: %s", exc.__class__.__name__, exc)
            self.tokenizer = None
            self.model = None

    def _initialize_state(self) -> None:
        """Initialize bot state variables."""
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
        self._validate_dependencies()
        logger.info("Loading model: %s", self.model_name)
        
        bnb_config = self._get_quantization_config()
        self._load_tokenizer()
        self._load_model(bnb_config)
        logger.info("Model loaded successfully!")

    def _validate_dependencies(self) -> None:
        """Validate required dependencies are available."""
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers not available")
        if torch is None:
            raise ImportError("torch not available")

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config based on device."""
        if self.device == "cuda":
            logger.info("Setting up 4-bit quantization config for CUDA")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.device == "mps":
            logger.info("MPS will use dynamic quantization after model loading")
        else:
            logger.info("No quantization config for %s", self.device)
        return None

    def _apply_mps_quantization(self) -> None:
        """Apply dynamic quantization optimized for MPS."""
        try:
            logger.info("Applying dynamic quantization for MPS...")
            # Apply dynamic quantization to linear layers
            from torch.quantization import quantize_dynamic
            
            # Quantize the model's linear layers
            self.model = quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Dynamic quantization applied successfully for MPS")
        except Exception as e:
            logger.warning("Could not apply dynamic quantization: %s", e)
            logger.info("Continuing without quantization...")

    def _load_tokenizer(self) -> None:
        """Load and configure tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")

    def _load_model(self, bnb_config: Optional[BitsAndBytesConfig]) -> None:
        """Load model with appropriate configuration."""
        logger.info("Loading model (this may take several minutes)...")
        if self.device == "cuda" and bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif self.device == "mps":
            # MPS (Apple Silicon) configuration with quantization
            logger.info("Loading model for MPS with dynamic quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=None,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)
            # Apply dynamic quantization for MPS
            self._apply_mps_quantization()
        else:
            # CPU fallback
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None
            )

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
        
        try:
            prompt = self._create_interpretation_prompt(chart_json)
            response = self._generate_llm_response(prompt)
            interpretation = self._extract_interpretation(response, prompt)
            
            logger.info("LLM interpretation completed successfully")
            return {
                "interpretation": interpretation,
                "chart_data": json.loads(chart_json)
            }
        except Exception as e:
            logger.error("Error during LLM interpretation: %s", e)
            # Return a fallback interpretation
            return {
                "interpretation": f"Chart generated successfully but interpretation failed: {e}",
                "chart_data": json.loads(chart_json)
            }

    def _create_interpretation_prompt(self, chart_json: str) -> str:
        """Create the prompt for chart interpretation."""
        # Parse chart data to extract key elements
        try:
            chart_data = json.loads(chart_json)
            
            # Extract key astrological elements
            sun_sign = chart_data.get("planets", {}).get("Sun", {}).get("sign", "Unknown")
            moon_sign = chart_data.get("planets", {}).get("Moon", {}).get("sign", "Unknown")
            rising_sign = chart_data.get("angles", {}).get("Asc", {}).get("sign", "Unknown")
            
            # Get a few key planets
            key_planets = []
            for planet in ["Mercury", "Venus", "Mars"]:
                planet_data = chart_data.get("planets", {}).get(planet, {})
                if planet_data:
                    key_planets.append(f"{planet} in {planet_data.get('sign', 'Unknown')}")
            
            # Get a few key aspects
            aspects = chart_data.get("aspects", [])[:3]  # Just first 3 aspects
            key_aspects = [f"{asp.get('planet1', '')} {asp.get('type', '')} {asp.get('planet2', '')}" for asp in aspects]
            
            # Create a concise summary
            chart_summary = f"""Sun: {sun_sign}
Moon: {moon_sign}  
Rising: {rising_sign}
Key Planets: {', '.join(key_planets)}
Key Aspects: {', '.join(key_aspects)}"""
            
        except Exception as e:
            # Fallback if parsing fails
            chart_summary = str(chart_json)[:500] + "..."
        
        return f"""<|system|>
You are an expert astrologer specializing in personality analysis through birth charts. Provide insightful, specific interpretations based on astrological placements.
<|end|>
<|user|>
Analyze this birth chart and provide a personality interpretation:

{chart_summary}

Provide a structured analysis covering:
1. Core personality traits (based on Sun, Moon, Rising)
2. Strengths and talents
3. Communication style (Mercury influence)
4. Relationship patterns (Venus influence)
5. Drive and ambition (Mars influence)

Keep it concise but insightful, focusing on practical personality insights.
<|end|>
<|assistant|>"""

    def _generate_llm_response(self, prompt: str) -> str:
        """Generate response from the language model."""
        logger.info("Tokenizing and generating response...")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move inputs to the same device as model
        if self.device in ["cuda", "mps"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        logger.info("Input tokens: %d", inputs['input_ids'].shape[1])
        logger.info("Starting generation on device: %s", self.device)
        
        try:
            with torch.no_grad():
                # Optimized settings for astrological interpretation
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # Increased for full interpretation
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    early_stopping=True,
                    num_beams=1,  # Greedy for speed
                )
            
            logger.info("Generation completed, decoding...")
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("Generated response length: %d characters", len(full_response))
            return full_response
            
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return f"Generation failed: {str(e)}"

    def _extract_interpretation(self, response: str, prompt: str) -> str:
        """Extract the interpretation from the model response."""
        logger.info("Extracting interpretation from response...")
        
        # Try to find the assistant delimiter for Phi-3.5
        if "<|assistant|>" in response:
            interpretation = response.split("<|assistant|>")[-1].strip()
            logger.info("Found <|assistant|> delimiter, extracted %d chars", len(interpretation))
        elif "[/INST]" in response:
            # Fallback for Mistral format
            interpretation = response.split("[/INST]")[-1].strip()
            logger.info("Found [/INST] delimiter, extracted %d chars", len(interpretation))
        else:
            # Fallback: try to remove the prompt
            interpretation = response[len(prompt):].strip()
            logger.info("No delimiter found, removed prompt, extracted %d chars", len(interpretation))
        
        # If still empty, return the full response as fallback
        if not interpretation:
            logger.warning("Interpretation is empty, returning full response")
            interpretation = response.strip()
        
        return interpretation

    def test_model(self) -> str:
        """Test the model with a simple prompt."""
        logger.info("Testing model with simple prompt...")
        if self.tokenizer is None or self.model is None:
            return "Model not loaded"
        
        test_prompt = "<|system|>\nYou are a helpful assistant.\n<|end|>\n<|user|>\nSay hello!\n<|end|>\n<|assistant|>"
        try:
            response = self._generate_llm_response(test_prompt)
            return self._extract_interpretation(response, test_prompt)
        except Exception as e:
            return f"Test failed: {e}"

    def generate_birth_chart(self, birth_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate birth chart with optional hardcoded birth data."""
        logger.info("Starting birth chart generation...")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "Language model not loaded. Install required packages and re-run."
            )
        
        # First test the model
        logger.info("Testing model first...")
        test_result = self.test_model()
        logger.info("Model test result: %s", test_result[:100])
        
        # Use provided birth data or generate random
        if birth_data is None:
            logger.info("Generating random birth data...")
            birth_data = self.generate_random_birth_data()
        else:
            logger.info("Using provided birth data: %s", birth_data)
        
        logger.info("Creating chart JSON...")
        self.current_chart_json = self.generate_chart_json(birth_data)
        
        try:
            chart_data = json.loads(self.current_chart_json)
            logger.info("Chart JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.error("Error parsing chart JSON: %s", e)
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

    def get_sample_birth_data(self) -> Dict[str, Any]:
        """Get sample birth data for testing."""
        # You can modify these values for different test cases
        return {
            "date": "2000-10-11",  # July 15, 1990 (Cancer Sun)
            "time": "11:59",       # 2:30 PM
            "lat": 25.44,        # New York City latitude
            "lon": 81.83,       # New York City longitude
        }
        
        # Other interesting examples to try:
        # Leo Sun: {"date": "1985-08-10", "time": "12:00", "lat": 34.0522, "lon": -118.2437}  # LA
        # Scorpio Sun: {"date": "1992-11-08", "time": "18:45", "lat": 51.5074, "lon": -0.1278}  # London
        # Aquarius Sun: {"date": "1988-02-14", "time": "09:15", "lat": 48.8566, "lon": 2.3522}  # Paris

    def get_personality(self) -> Optional[str]:
        """Get the current personality interpretation."""
        if self.current_personality:
            return self.current_personality["interpretation"]
        return None

if __name__ == "__main__":
    bot = AstroPersonalityBot()
    print(f"setup bot...done!")
