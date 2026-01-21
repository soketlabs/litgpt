

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# MODEL_ID = "soketlabs/saarthi-agri-v1"


# def main():
#     print("\n================ LOADING MODEL ================\n")


#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_ID,
#         use_fast=True,
#         trust_remote_code=True
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     model.eval()

#     # Manually tie weights if they are not tied (Gemma3 specific)
#     #print("Manually tying lm_head weights to embed_tokens...")
#     # model.lm_head.weight = model.model.language_model.embed_tokens.weight

#     print("✅ Model loaded successfully\n")

#     # -------------------------
#     # Chat messages
#     # -------------------------
#     messages = [{
#                 "role": "system",
#                 "content": """You are a helpful agronomy expert providing crop advisory to farmers based on location and weather conditions.

#             Output Order (Mandatory and Strict):
#             1. The very first output token must be exactly <unused0>
#             2. Immediately after <unused0>, produce a structured analytical reasoning section in English covering:
#             - Crop suitability for the given month and region
#             - Climate and rainfall assessment
#             - Soil and irrigation considerations
#             - Risk factors (pests, diseases, weather stress)
#             - Recommended agronomic practices
#             3. After the reasoning is complete, output exactly <unused1>
#             4. Only after <unused1>, produce the final advisory response intended for the user.

#             Output Restrictions:
#             - Do not output anything before <unused0>.
#             - Do not output anything between <unused0> and <unused1> except the analytical reasoning section.
#             - Do not repeat <unused0> or <unused1>.
#             - Do not include meta commentary or explanations about the protocol.
#             - The final advisory must be written strictly in the language requested by the user.
#             """
#             },

#         {
#             "role": "user",
#             "content": (
#                 "I want crop advisory for varanasi. "
#                 "Its jan and I am thinking of sowing sugercase. "
#                 "Rain has been very fragmented. "
#                 "Think hard and give me advisory in hindi"
#             )
#         }
#     ]

#     # -------------------------
#     # Apply chat template
#     # -------------------------
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     print("\n================ PROMPT ================\n")
#     print(prompt)

#     # -------------------------
#     # Tokenize
#     # -------------------------
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     # -------------------------
#     # Streamer (live output)
#     # -------------------------
#     streamer = TextStreamer(
#         tokenizer,
#         skip_prompt=True,
#         skip_special_tokens=True
#     )

#     # -------------------------
#     # Generate
#     # -------------------------
#     print("\n================ MODEL OUTPUT ================\n")

#     with torch.no_grad():
#         model.generate(
#             **inputs,
#             max_new_tokens=5000,
#             do_sample=True,
#             temperature=0.8,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             streamer=streamer
#         )

#     print("\n\n✅ Inference complete\n")


# if __name__ == "__main__":
#     main()



#################### more specific inference script ####################

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_ID = "soketlabs/saarthi-agri-v1"

# -------------------------
# Structured Input Data
# -------------------------
INPUT_DATA = {
    "Month": "January",
    "Growth Stage": "फूल आना",
    "Weather": "गरम और आर्द्र मौसम",
    "Rainfall_mm": 42,          # numeric (mm)
    "Humidity_percent": 78,  
    "Pressure_hPa": 1018,  # numeric (%)
    "Soil Type": "काली मिट्टी",
    "Farming Practice": "सामान्य खेती",
    "Region": "महाराष्ट्र",
    "Language": "Hindi",
    "Crop": "कपास",
    "Stress": "सफेद मक्खी",
}

# INPUT_DATA = {
#     "Month": "February",
#     "Growth Stage": "फल बनना",
#     "Weather": "हल्का गर्म और शुष्क मौसम",
#     "Rainfall_mm": 18,          
#     "Humidity_percent": 52,  
#     "Pressure_hPa": 1012,  
#     "Soil Type": "लाल दोमट मिट्टी",
#     "Farming Practice": "ड्रिप सिंचाई",
#     "Region": "कर्नाटक",
#     "Language": "Hindi",
#     "Crop": "टमाटर",
#     "Stress": "फलों में फटने की समस्या",
# }

# INPUT_DATA = {
#     "Month": "February",
#     "Growth Stage": "Flowering Stage",
#     "Weather": "Mild warm and dry weather",
#     "Rainfall_mm": 18,
#     "Humidity_percent": 52,
#     "Pressure_hPa": 1012,
#     "Soil Type": "Red loam soil",
#     "Farming Practice": "Drip irrigation",
#     "Region": "Karnataka",
#     "Language": "English",
#     "Crop": "Tomato",
#     "Stress": "Fruit cracking problem"
# }



def main():
    print("\n================ LOADING MODEL ================\n")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("✅ Model loaded successfully\n")

    # -------------------------
    # Build User Input Block
    # -------------------------
    user_input_block = (
        f"Month: {INPUT_DATA['Month']}\n"
        f"Growth Stage: {INPUT_DATA['Growth Stage']}\n"
        f"Weather: {INPUT_DATA['Weather']}\n"
        f"Rainfall: {INPUT_DATA['Rainfall_mm']} mm\n"
        f"Humidity: {INPUT_DATA['Humidity_percent']} %\n"
        f"Pressure: {INPUT_DATA['Pressure_hPa']} hPa\n"   
        f"Soil Type: {INPUT_DATA['Soil Type']}\n"
        f"Farming Practice: {INPUT_DATA['Farming Practice']}\n"
        f"Region: {INPUT_DATA['Region']}\n"
        f"Language: {INPUT_DATA['Language']}\n"
        f"Crop: {INPUT_DATA['Crop']}\n"
        f"Stress: {INPUT_DATA['Stress']}\n"
    )


    messages = [
        {
            "role": "system",
            "content": """You are a helpful District Agricultural Officer  providing crop advisory to farmers based on location and various climatic conditions given as input.

Output Order (Mandatory and Strict):
1. The very first output token must be exactly <unused0>
2. Immediately after <unused0>, produce a structured analytical reasoning section in English covering:
- Crop suitability for the given Month and Region, considering the Crop type and Growth Stage
- Climate assessment using Weather description, Rainfall_mm, Humidity_percent, and Pressure_hPa
- Soil behavior, soil moisture retention, and irrigation needs based on Soil Type and Rainfall_mm
- Growth-stage-specific agronomic requirements and timing considerations
- Risk analysis including Stress factors (pests/diseases), humidity-driven disease risk, and weather-related stress
- Impact of Farming Practice on productivity and risk mitigation
- Integrated recommendation logic combining all above parameters coherently
3. After the reasoning is complete, output exactly <unused1>
4. Only after <unused1>, produce the final advisory response intended for the user.

Output Restrictions:
- Do not output anything before <unused0>.
- Do not output anything between <unused0> and <unused1> except the analytical reasoning section.
- Do not repeat <unused0> or <unused1>.
- Do not include meta commentary or explanations about the protocol.
- The final advisory must be written strictly in the language requested by the user.
"""
        },
        {
            "role": "user",
            "content": (
                "Please generate crop advisory using the following structured input:\n\n"
                f"{user_input_block}\n"
                "Think carefully and follow the output protocol strictly."
            )
        }
    ]

    # -------------------------
    # Apply chat template
    # -------------------------
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n================ PROMPT ================\n")
    print(prompt)

    # -------------------------
    # Tokenize
    # -------------------------
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # -------------------------
    # Streamer (live output)
    # -------------------------
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # -------------------------
    # Generate
    # -------------------------
    print("\n================ MODEL OUTPUT ================\n")

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5000,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            streamer=streamer
        )

    print("\n\n✅ Inference complete\n")


if __name__ == "__main__":
    main()
