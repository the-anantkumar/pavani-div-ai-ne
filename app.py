"""Streamlined Astrological Personality Chatbot.

This application generates an astrological chart using the `immanuel` library,
feeds the JSON data to a language model for interpretation and provides a chat
interface via Gradio or FastAPI.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from immanuel import charts
from immanuel.const import chart, calc
from immanuel.setup import settings



class AstroPersonalityBot:
    """Main chatbot class handling chart generation and conversations."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self._setup_immanuel()
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_llm()
        self.current_chart_json: Optional[str] = None
        self.current_personality: Optional[Dict[str, Any]] = None
        self.chat_history: List[Tuple[str, str]] = []

    def _setup_immanuel(self) -> None:
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
        birth_datetime = datetime.combine(
            birth_date.date(),
            datetime.min.time(),
        ) + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        lat = random.uniform(-90, 90)
        lon = random.uniform(-180, 180)
        return {
            "date": birth_datetime.strftime("%Y-%m-%d"),
            "time": birth_datetime.strftime("%H:%M"),
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "city": "Random City",
            "country": "Random Country",
        }

    def generate_chart_json(self, birth_data: Dict[str, Any]) -> str:
        natal = charts.Natal(birth_data)
        chart_dict: Dict[str, Any] = {
            "birth_data": birth_data,
            "planets": {},
            "houses": {},
            "aspects": [],
            "angles": {},
        }
        for obj_name, obj_data in natal.objects.items():
            if hasattr(obj_data, "sign"):
                chart_dict["planets"][obj_name] = {
                    "sign": obj_data.sign.name,
                    "degree": float(obj_data.lon),
                    "house": obj_data.house.number if hasattr(obj_data, "house") and obj_data.house else None,
                    "retrograde": getattr(obj_data, "retrograde", False),
                }
        for house_num, house_data in natal.houses.items():
            chart_dict["houses"][f"House_{house_num}"] = {
                "sign": house_data.sign.name,
                "degree": float(house_data.lon),
            }
        for angle_name, angle_data in natal.angles.items():
            chart_dict["angles"][angle_name] = {
                "sign": angle_data.sign.name,
                "degree": float(angle_data.lon),
            }
        for aspect in natal.aspects:
            chart_dict["aspects"].append(
                {
                    "planet1": aspect.objects[0].name,
                    "planet2": aspect.objects[1].name,
                    "type": aspect.type.name,
                    "angle": float(aspect.angle),
                    "orb": float(aspect.orb),
                }
            )
        return json.dumps(chart_dict, indent=2)

    def interpret_chart_to_personality(self, chart_json: str) -> str:
        interpretation_prompt = (
            "<s>[INST] You are an expert astrologer. Analyze this birth chart JSON "
            "and create a detailed personality profile.\n\nBirth Chart Data:\n"
            f"{chart_json}\n\nBased on this astrological data, provide:\n"
            "1. A comprehensive personality overview (2-3 paragraphs)\n"
            "2. Communication style\n"
            "3. Emotional patterns\n"
            "4. Strengths and challenges\n"
            "5. How this person would interact in conversations\n\n"
            "Focus on creating a vivid, unique personality that feels authentic "
            "and three-dimensional. Be specific about behavioral patterns, speech "
            "patterns, interests, and worldview based on the astrological placements.\n\n"
            "End with a section titled \"PERSONALITY ESSENCE:\" that summarizes the "
            "key traits in 2-3 sentences that can be used as a character description. [/INST]"
        )
        inputs = self.tokenizer(interpretation_prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        interpretation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        interpretation = interpretation.split("[/INST]")[-1].strip()
        return interpretation

    def create_system_prompt(self, personality_description: str) -> str:
        if "PERSONALITY ESSENCE:" in personality_description:
            essence = personality_description.split("PERSONALITY ESSENCE:")[-1].strip()
        else:
            essence = personality_description.split("\n")[0]
        system_prompt = (
            "You are a person with the following personality profile:\n\n"
            f"{essence}\n\n"
            "Important guidelines:\n"
            "- Embody this personality naturally in all responses\n"
            "- Use speech patterns and communication style described above\n"
            "- React to topics based on your astrological makeup\n"
            "- Be helpful while maintaining your unique personality\n"
            "- Don't mention astrology unless specifically asked\n"
            "- Express opinions and preferences aligned with your character\n"
            "- Show emotional responses consistent with your profile"
        )
        return system_prompt

    def generate_new_personality(self) -> Tuple[str, str, Dict[str, Any]]:
        birth_data = self.generate_random_birth_data()
        self.current_chart_json = self.generate_chart_json(birth_data)
        chart_data = json.loads(self.current_chart_json)
        interpretation = self.interpret_chart_to_personality(self.current_chart_json)
        self.current_personality = {
            "interpretation": interpretation,
            "system_prompt": self.create_system_prompt(interpretation),
            "birth_data": birth_data,
            "chart_data": chart_data,
        }
        self.chat_history = []
        display_text = (
            "### \U0001F31F New Personality Generated!\n\n"
            f"**Birth Data:**\n- \U0001F4C5 {birth_data['date']} at {birth_data['time']}\n"
            f"- \U0001F4CD {birth_data['city']}, {birth_data['country']}\n\n"
            "**Key Placements:**\n"
            f"- \u2600\ufe0f Sun: {chart_data['planets']['Sun']['sign']}\n"
            f"- \U0001F319 Moon: {chart_data['planets']['Moon']['sign']}\n"
            f"- \u2B06\uFE0F Ascendant: {chart_data['angles']['Asc']['sign']}\n\n"
            "**Personality Profile:**\n"
            f"{interpretation}"
        )
        return display_text, self.current_chart_json, chart_data

    def chat(self, message: str) -> str:
        if not self.current_personality:
            return "Please generate a personality first by clicking the button above!"
        chat_prompt = (
            f"<s>[INST] <<SYS>>\n{self.current_personality['system_prompt']}\n<</SYS>>\n\n{message} [/INST]"
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()
        return response

    def create_gradio_interface(self) -> gr.Blocks:
        with gr.Blocks(theme=gr.themes.Soft(), title="Astrological Personality Chatbot") as interface:
            gr.Markdown(
                "# \U0001F31F Astrological Personality Chatbot\n"
                "*Generate a unique personality from astrological data and chat with them!*"
            )
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        generate_btn = gr.Button("\U0001F3B2 Generate New Personality", variant="primary", size="lg")
                        personality_display = gr.Markdown("Click 'Generate' to create a personality!")
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(height=500, label="Conversation")
                        msg = gr.Textbox(placeholder="Type your message here...", label="Your Message", lines=2)
                        with gr.Row():
                            submit = gr.Button("Send", variant="primary")
                            clear = gr.Button("Clear Chat")
            with gr.Tab("Chart Data"):
                gr.Markdown("### \U0001F4CA Raw Astrological Data")
                chart_json_display = gr.JSON(label="Immanuel Chart JSON")
                chart_data_display = gr.JSON(label="Parsed Chart Data")
            current_personality_state = gr.State(None)

            def generate_and_update():
                display, json_str, data = self.generate_new_personality()
                return display, json_str, data, self.current_personality, []

            def chat_response(message, history, personality_state):
                if not personality_state:
                    bot_response = "Please generate a personality first!"
                else:
                    if self.current_personality != personality_state:
                        self.current_personality = personality_state
                    bot_response = self.chat(message)
                history.append((message, bot_response))
                return "", history

            generate_btn.click(
                fn=generate_and_update,
                outputs=[personality_display, chart_json_display, chart_data_display, current_personality_state, chatbot],
            )
            msg.submit(
                fn=chat_response,
                inputs=[msg, chatbot, current_personality_state],
                outputs=[msg, chatbot],
            )
            submit.click(
                fn=chat_response,
                inputs=[msg, chatbot, current_personality_state],
                outputs=[msg, chatbot],
            )
            clear.click(fn=lambda: [], outputs=[chatbot])
        return interface


app = FastAPI(title="Astrological Personality API")
bot: Optional[AstroPersonalityBot] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class PersonalityRequest(BaseModel):
    birth_data: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    global bot
    bot = AstroPersonalityBot()


@app.post("/api/generate_personality")
async def generate_personality(request: PersonalityRequest):
    if request.birth_data:
        chart_json = bot.generate_chart_json(request.birth_data)
    else:
        birth_data = bot.generate_random_birth_data()
        chart_json = bot.generate_chart_json(birth_data)
    interpretation = bot.interpret_chart_to_personality(chart_json)
    bot.current_personality = {
        "interpretation": interpretation,
        "system_prompt": bot.create_system_prompt(interpretation),
        "chart_json": chart_json,
    }
    return {
        "success": True,
        "chart": json.loads(chart_json),
        "interpretation": interpretation,
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not bot.current_personality:
        raise HTTPException(status_code=400, detail="No personality generated yet")
    response = bot.chat(request.message)
    return {"response": response, "session_id": request.session_id}


if __name__ == "__main__":
    bot = AstroPersonalityBot()
    interface = bot.create_gradio_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
    # For FastAPI alternative, uncomment the following line:
    # uvicorn.run(app, host="0.0.0.0", port=8000)
