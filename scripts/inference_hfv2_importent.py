#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_ID = "SayantanJoker/saarthi-v1-untie"


def main():
    print("\n================ LOADING MODEL ================\n")

    # -------------------------
    # Load tokenizer + model
    # -------------------------
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
        # tie_word_embeddings=True
    )
    model.eval()

    # Manually tie weights if they are not tied (Gemma3 specific)
    #print("Manually tying lm_head weights to embed_tokens...")
    # model.lm_head.weight = model.model.language_model.embed_tokens.weight

    print("✅ Model loaded successfully\n")

    # -------------------------
    # Chat messages
    # -------------------------
    messages = [
        {
            "role": "system",
            "content": "You are a helpful agronomy expert providing crop advisory to farmers based on location and weather conditions."
        },
        {
            "role": "user",
            "content": (
                "I want crop advisory for varanasi. "
                "Its jan and I am thinking of sowing sugercase. "
                "Rain has been very fragmented. "
                "Think hard and give me advisory in hindi"
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