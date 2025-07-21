# Streamlined Astrological Chatbot - Direct LLM Interpretation

## Core Concept
1. Generate astrological chart with Immanuel → JSON
2. Pass JSON to LLM → Get personality interpretation
3. Use personality as system prompt → Chat with that personality

## Project Structure (Simplified)
```
astro-chatbot-simple/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── data/
│   └── locations.csv     # City database
└── utils/
    └── chart_to_text.py  # Helper to format chart data
```

## Complete Implementation

### 1. Main Application (app.py)
```python
"""
Streamlined Astrological Personality Chatbot
Uses Immanuel for chart generation and LLM for interpretation
"""

import json
import random
import gradio as gr
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import pandas as pd
from immanuel import charts
from immanuel.const import chart
from immanuel.setup import settings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class AstroPersonalityBot:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """Initialize the chatbot with Immanuel and LLM."""
        
        # Configure Immanuel
        self._setup_immanuel()
        
        # Load locations
        self.locations = pd.read_csv('data/locations.csv')
        
        # Setup LLM
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_llm()
        
        # Session state
        self.current_chart_json = None
        self.current_personality = None
        self.chat_history = []
        
    def _setup_immanuel(self):
        """Configure Immanuel settings."""
        settings.house_system = chart.PLACIDUS
        settings.objects = [
            chart.SUN, chart.MOON, chart.MERCURY, chart.VENUS,
            chart.MARS, chart.JUPITER, chart.SATURN,
            chart.URANUS, chart.NEPTUNE, chart.PLUTO,
            chart.ASC, chart.MC, chart.DESC, chart.IC
        ]
        settings.aspects = [
            chart.CONJUNCTION, chart.OPPOSITION,
            chart.SQUARE, chart.TRINE, chart.SEXTILE
        ]
        settings.default_orb = 8.0
    
    def _setup_llm(self):
        """Initialize the LLM with quantization."""
        print(f"Loading {self.model_name}...")
        
        # Quantization config for memory efficiency
        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        print("Model loaded successfully!")
    
    def generate_random_birth_data(self) -> Dict[str, Any]:
        """Generate random birth data."""
        # Random date between 1950-2005
        start_date = datetime(1950, 1, 1)
        end_date = datetime(2005, 12, 31)
        random_days = random.randint(0, (end_date - start_date).days)
        birth_date = start_date + timedelta(days=random_days)
        
        # Random time
        birth_datetime = datetime.combine(
            birth_date.date(),
            datetime.min.time()
        ) + timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Random location
        location = self.locations.sample(1).iloc[0]
        
        return {
            'date': birth_datetime.strftime('%Y-%m-%d'),
            'time': birth_datetime.strftime('%H:%M'),
            'lat': location['latitude'],
            'lon': location['longitude'],
            'city': location['city'],
            'country': location['country']
        }
    
    def generate_chart_json(self, birth_data: Dict) -> str:
        """Generate Immanuel chart and return as JSON."""
        # Create natal chart
        natal = charts.Natal(birth_data)
        
        # Convert to dictionary for JSON serialization
        chart_dict = {
            'birth_data': birth_data,
            'planets': {},
            'houses': {},
            'aspects': [],
            'angles': {}
        }
        
        # Extract planetary positions
        for obj_name, obj_data in natal.objects.items():
            if hasattr(obj_data, 'sign'):
                chart_dict['planets'][obj_name] = {
                    'sign': obj_data.sign.name,
                    'degree': float(obj_data.lon),
                    'house': obj_data.house.number if hasattr(obj_data, 'house') and obj_data.house else None,
                    'retrograde': obj_data.retrograde if hasattr(obj_data, 'retrograde') else False
                }
        
        # Extract houses
        for house_num, house_data in natal.houses.items():
            chart_dict['houses'][f'House_{house_num}'] = {
                'sign': house_data.sign.name,
                'degree': float(house_data.lon)
            }
        
        # Extract angles
        for angle_name, angle_data in natal.angles.items():
            chart_dict['angles'][angle_name] = {
                'sign': angle_data.sign.name,
                'degree': float(angle_data.lon)
            }
        
        # Extract aspects
        for aspect in natal.aspects:
            chart_dict['aspects'].append({
                'planet1': aspect.objects[0].name,
                'planet2': aspect.objects[1].name,
                'type': aspect.type.name,
                'angle': float(aspect.angle),
                'orb': float(aspect.orb)
            })
        
        return json.dumps(chart_dict, indent=2)
    
    def interpret_chart_to_personality(self, chart_json: str) -> str:
        """Use LLM to interpret chart JSON into personality description."""
        
        interpretation_prompt = f"""<s>[INST] You are an expert astrologer. Analyze this birth chart JSON and create a detailed personality profile.

Birth Chart Data:
{chart_json}

Based on this astrological data, provide:
1. A comprehensive personality overview (2-3 paragraphs)
2. Communication style
3. Emotional patterns
4. Strengths and challenges
5. How this person would interact in conversations

Focus on creating a vivid, unique personality that feels authentic and three-dimensional. Be specific about behavioral patterns, speech patterns, interests, and worldview based on the astrological placements.

End with a section titled "PERSONALITY ESSENCE:" that summarizes the key traits in 2-3 sentences that can be used as a character description. [/INST]"""

        # Generate interpretation
        inputs = self.tokenizer(interpretation_prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        interpretation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        interpretation = interpretation.split("[/INST]")[-1].strip()
        
        return interpretation
    
    def create_system_prompt(self, personality_description: str) -> str:
        """Extract personality essence and create system prompt."""
        
        # Extract the personality essence section
        if "PERSONALITY ESSENCE:" in personality_description:
            essence = personality_description.split("PERSONALITY ESSENCE:")[-1].strip()
        else:
            # Fallback: use first paragraph
            essence = personality_description.split('\n')[0]
        
        system_prompt = f"""You are a person with the following personality profile:

{essence}

Important guidelines:
- Embody this personality naturally in all responses
- Use speech patterns and communication style described above
- React to topics based on your astrological makeup
- Be helpful while maintaining your unique personality
- Don't mention astrology unless specifically asked
- Express opinions and preferences aligned with your character
- Show emotional responses consistent with your profile

Respond naturally as this person would, with all their quirks, interests, and communication patterns."""

        return system_prompt
    
    def generate_new_personality(self) -> Tuple[str, str, Dict]:
        """Generate new personality and return display info."""
        
        # Generate random birth data
        birth_data = self.generate_random_birth_data()
        
        # Generate chart JSON
        self.current_chart_json = self.generate_chart_json(birth_data)
        chart_data = json.loads(self.current_chart_json)
        
        # Get LLM interpretation
        interpretation = self.interpret_chart_to_personality(self.current_chart_json)
        
        # Create system prompt from interpretation
        self.current_personality = {
            'interpretation': interpretation,
            'system_prompt': self.create_system_prompt(interpretation),
            'birth_data': birth_data,
            'chart_data': chart_data
        }
        
        # Reset chat history
        self.chat_history = []
        
        # Create display text
        display_text = f"""
### 🌟 New Personality Generated!

**Birth Data:**
- 📅 {birth_data['date']} at {birth_data['time']}
- 📍 {birth_data['city']}, {birth_data['country']}

**Key Placements:**
- ☀️ Sun: {chart_data['planets']['Sun']['sign']}
- 🌙 Moon: {chart_data['planets']['Moon']['sign']}
- ⬆️ Ascendant: {chart_data['angles']['Asc']['sign']}

**Personality Profile:**
{interpretation}
"""
        
        return display_text, self.current_chart_json, chart_data
    
    def chat(self, message: str) -> str:
        """Generate response based on current personality."""
        
        if not self.current_personality:
            return "Please generate a personality first by clicking the button above!"
        
        # Create chat prompt
        chat_prompt = f"""<s>[INST] <<SYS>>
{self.current_personality['system_prompt']}
<</SYS>>

{message} [/INST]"""
        
        # Generate response
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()
        
        return response
    
    def create_gradio_interface(self):
        """Create the Gradio UI."""
        
        with gr.Blocks(theme=gr.themes.Soft(), title="Astrological Personality Chatbot") as interface:
            gr.Markdown(
                """
                # 🌟 Astrological Personality Chatbot
                *Generate a unique personality from astrological data and chat with them!*
                """
            )
            
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        generate_btn = gr.Button("🎲 Generate New Personality", variant="primary", size="lg")
                        personality_display = gr.Markdown("Click 'Generate' to create a personality!")
                        
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(height=500, label="Conversation")
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            label="Your Message",
                            lines=2
                        )
                        with gr.Row():
                            submit = gr.Button("Send", variant="primary")
                            clear = gr.Button("Clear Chat")
            
            with gr.Tab("Chart Data"):
                gr.Markdown("### 📊 Raw Astrological Data")
                chart_json_display = gr.JSON(label="Immanuel Chart JSON")
                chart_data_display = gr.JSON(label="Parsed Chart Data")
            
            # State management
            current_personality_state = gr.State(None)
            
            # Event handlers
            def generate_and_update():
                display, json_str, data = self.generate_new_personality()
                return display, json_str, data, self.current_personality, []
            
            def chat_response(message, history, personality_state):
                if not personality_state:
                    bot_response = "Please generate a personality first!"
                else:
                    # Restore personality if needed
                    if self.current_personality != personality_state:
                        self.current_personality = personality_state
                    bot_response = self.chat(message)
                
                history.append((message, bot_response))
                return "", history
            
            # Connect events
            generate_btn.click(
                fn=generate_and_update,
                outputs=[personality_display, chart_json_display, chart_data_display, current_personality_state, chatbot]
            )
            
            msg.submit(
                fn=chat_response,
                inputs=[msg, chatbot, current_personality_state],
                outputs=[msg, chatbot]
            )
            
            submit.click(
                fn=chat_response,
                inputs=[msg, chatbot, current_personality_state],
                outputs=[msg, chatbot]
            )
            
            clear.click(
                fn=lambda: [],
                outputs=[chatbot]
            )
            
        return interface

# FastAPI Alternative Implementation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Astrological Personality API")
bot = None  # Will be initialized on startup

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
    """Generate new personality from birth data or random."""
    
    if request.birth_data:
        chart_json = bot.generate_chart_json(request.birth_data)
    else:
        birth_data = bot.generate_random_birth_data()
        chart_json = bot.generate_chart_json(birth_data)
    
    # Get interpretation
    interpretation = bot.interpret_chart_to_personality(chart_json)
    
    # Create personality
    bot.current_personality = {
        'interpretation': interpretation,
        'system_prompt': bot.create_system_prompt(interpretation),
        'chart_json': chart_json
    }
    
    return {
        'success': True,
        'chart': json.loads(chart_json),
        'interpretation': interpretation
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with current personality."""
    
    if not bot.current_personality:
        raise HTTPException(status_code=400, detail="No personality generated yet")
    
    response = bot.chat(request.message)
    
    return {
        'response': response,
        'session_id': request.session_id
    }

# Main entry point
if __name__ == "__main__":
    # For Gradio UI
    bot = AstroPersonalityBot()
    interface = bot.create_gradio_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
    
    # For FastAPI (uncomment to use)
    # uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Requirements File (requirements.txt)
```
# Core dependencies
immanuel>=1.0.0
pandas>=1.5.0
numpy>=1.24.0

# LLM dependencies
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
bitsandbytes>=0.41.0

# UI dependencies
gradio>=4.0.0
fastapi>=0.100.0
uvicorn>=0.23.0

# Utilities
python-dateutil>=2.8.0
pytz>=2023.3
```

### 3. Sample Locations File (data/locations.csv)
```csv
city,country,latitude,longitude
New York,USA,40.7128,-74.0060
Los Angeles,USA,34.0522,-118.2437
Chicago,USA,41.8781,-87.6298
London,UK,51.5074,-0.1278
Paris,France,48.8566,2.3522
Berlin,Germany,52.5200,13.4050
Tokyo,Japan,35.6762,139.6503
Mumbai,India,19.0760,72.8777
Sydney,Australia,-33.8688,151.2093
São Paulo,Brazil,-23.5505,-46.6333
Cairo,Egypt,30.0444,31.2357
Moscow,Russia,55.7558,37.6173
Dubai,UAE,25.2048,55.2708
Singapore,Singapore,1.3521,103.8198
Toronto,Canada,43.6532,-79.3832
```

## Usage Examples

### 1. Basic Usage
```python
# Initialize the bot
bot = AstroPersonalityBot()

# Generate a personality
display_text, chart_json, chart_data = bot.generate_new_personality()

# Chat with the personality
response = bot.chat("What's your favorite way to spend a weekend?")
print(response)
```

### 2. Custom Birth Data
```python
# Use specific birth data
custom_birth = {
    'date': '1990-07-15',
    'time': '14:30',
    'lat': 40.7128,
    'lon': -74.0060,
    'city': 'New York',
    'country': 'USA'
}

chart_json = bot.generate_chart_json(custom_birth)
interpretation = bot.interpret_chart_to_personality(chart_json)
```

### 3. Analyzing the LLM's Interpretation Process
```python
# See how the LLM interprets different aspects
chart_json = bot.generate_chart_json(bot.generate_random_birth_data())
print("Chart JSON:")
print(chart_json)

interpretation = bot.interpret_chart_to_personality(chart_json)
print("\nLLM Interpretation:")
print(interpretation)
```

## Deployment Options

### Option 1: Local Gradio
```bash
python app.py
# Opens at http://localhost:7860
```

### Option 2: FastAPI Server
```python
# Modify main section in app.py:
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Option 3: Docker Container
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data

EXPOSE 7860

CMD ["python", "app.py"]
```

### Option 4: Cloud Deployment (Hugging Face Spaces)
```yaml
# Create app.py with the code above
# Create requirements.txt
# Push to Hugging Face Spaces
# It will auto-deploy with Gradio
```

## Key Advantages of This Approach

1. **Simplicity**: No complex rule mappings or personality synthesis logic
2. **Flexibility**: LLM interprets any chart configuration naturally
3. **Authenticity**: Each personality is unique based on the full chart
4. **Extensibility**: Easy to add more astrological factors
5. **Maintainability**: Less code to maintain, no hardcoded rules

## Tips for Better Results

1. **Prompt Engineering**: Adjust the interpretation prompt for different personality styles
2. **Model Selection**: Try different models (Llama 3, Mixtral) for varied interpretations
3. **Temperature Settings**: Higher temperature (0.8-0.9) for more creative personalities
4. **Chart Complexity**: Include more aspects and planets for richer personalities
5. **Fine-tuning**: Train on astrological interpretation texts for better accuracy

## Minimal Quick Start

```python
# Absolute minimal version
from immanuel import charts
import random

# Generate random chart
birth_data = {
    'date': f'{random.randint(1950,2005)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
    'time': f'{random.randint(0,23):02d}:{random.randint(0,59):02d}',
    'lat': random.uniform(-90, 90),
    'lon': random.uniform(-180, 180)
}

# Get chart
chart = charts.Natal(birth_data)

# Extract key info
print(f"Sun: {chart.objects['Sun'].sign.name}")
print(f"Moon: {chart.objects['Moon'].sign.name}")
print(f"Ascendant: {chart.angles['Asc'].sign.name}")

# Pass to LLM for interpretation
# ... your LLM code here
```