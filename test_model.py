#!/usr/bin/env python3
"""Simple model test to debug LLM output."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    chart_json = {
    "birth_data": {
        "date": "1964-09-20",
        "time": "23:50",
        "lat": -36.88584363207414,
        "lon": -90.30979080355354
    },
    "planets": {},
    "houses": {
        "House_1": {"sign": "Pisces", "degree": 357.5992508634227},
        "House_2": {"sign": "Aries", "degree": 24.131824263994087},
        "House_3": {"sign": "Taurus", "degree": 54.48387826705887},
        "House_4": {"sign": "Gemini", "degree": 87.32118567367439},
        "House_5": {"sign": "Leo", "degree": 120.20620786765028},
        "House_6": {"sign": "Virgo", "degree": 150.7292674700846},
        "House_7": {"sign": "Virgo", "degree": 177.59925086342264},
        "House_8": {"sign": "Libra", "degree": 204.13182426399408},
        "House_9": {"sign": "Scorpio", "degree": 234.48387826705886},
        "House_10": {"sign": "Sagittarius", "degree": 267.3211856736744},
        "House_11": {"sign": "Aquarius", "degree": 300.2062078676503},
        "House_12": {"sign": "Pisces", "degree": 330.7292674700846}
    },
    "aspects": [
        {"planet1": "Moon", "planet2": "Sun", "type": "Opposition", "angle": 9.0, "orb": 10.0},
        {"planet1": "Sun", "planet2": "Mars", "type": "Sextile", "angle": 5.0, "orb": 6.0},
        {"planet1": "Sun", "planet2": "Jupiter", "type": "Trine", "angle": 1.0, "orb": 10.0},
        {"planet1": "Desc", "planet2": "Sun", "type": "Conjunction", "angle": 0.0, "orb": 10.0}
    ],
    "angles": {
        "Asc": {"sign": "Pisces", "degree": 357.5992508634227},
        "MC": {"sign": "Sagittarius", "degree": 267.3211856736744},
        "Desc": {"sign": "Virgo", "degree": 177.5992508634227},
        "IC": {"sign": "Gemini", "degree": 87.32118567367439}
    }
}

    # Simple test prompt
    prompt = f"<s>[INST] You are an expert asrologer. Anaylyse this birth chart: {chart_json} [/INST]"
    
    print(f"Input prompt: {prompt}")
    print("-" * 50)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5000,  # Increased from 50 to 500 tokens
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Full response: {full_response}")
    print("-" * 50)
    
    # Extract just the answer
    if "[/INST]" in full_response:
        answer = full_response.split("[/INST]")[-1].strip()
        print(f"Extracted answer: {answer}")
    else:
        print("Could not find [/INST] delimiter")

if __name__ == "__main__":
    test_model()