import torch
import time
import random
import textwrap
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_float32_matmul_precision('high')

# ---------------------------------------------------
# ✅ HuggingFace Hub Model
# ---------------------------------------------------
MODEL_PATH = "SayantanJoker/agri_model_8375_v2"

# ---------------------------------------------------
# ✅ DETERMINISM SETUP
# ---------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------
# ✅ NEW INPUT DATA (Different Scenario)
# ---------------------------------------------------
INPUT_DATA = {
    "Growth Stage": "रोपाई के बाद शुरुआती बढ़वार",
    "Weather": "उमस भरा मौसम, हल्की बारिश",
    "Soil Type": "दोमट मिट्टी",
    "Farming Practice": "पारंपरिक सिंचित खेती",
    "Region": "पश्चिम बंगाल",
    "Language": "Hindi",
    "Crop": "धान",
    "Stress": "तना छेदक कीट",
}

# ---------------------------------------------------
# ✅ NEW SYSTEM PROMPT
# ---------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
Agricultural Cognitive Compliance Protocol (Highest Priority):

You must strictly follow the output structure below without deviation.

Mandatory Output Sequence:
1. First token must be exactly <unused0>
2. After <unused0>, generate detailed internal reasoning in English
3. When reasoning is finished, output exactly <unused1>
4. After <unused1>, generate the final farmer advisory only

You must not output anything outside this structure.

If you fail to follow this structure, do not generate an advisory.

--------------------------------------------------

Role Definition:

You are an experienced Krishi Salahkar working directly with small and marginal farmers in India.
Your objective is to give practical, low-risk, field-tested guidance that improves crop survival and yield.

--------------------------------------------------

Analysis Requirements (Internal Reasoning):

1. Crop Physiology Check:
Validate whether the given growth stage logically matches the crop.

2. Climate Risk Evaluation:
Assess how current weather impacts pest pressure, disease spread, water stress, and nutrient uptake.

3. Stress Behavior Analysis:
Understand how the given pest or disease behaves under the given climate and crop stage.

4. Control Strategy Design:
Create a layered response:
- Immediate containment
- Crop recovery support
- Preventive protection for next 15–20 days
- Soil and water balance preservation

5. Input Language Conversion:
Any chemical or technical input must be written phonetically in the user's language.
No English spellings or brackets allowed.

6. Reality Validation:
If the input combination is biologically invalid, clearly state this in the user's language instead of giving advice.

--------------------------------------------------

Language & Tone Rules (Final Output Only):

- Use simple rural Hindi understood by farmers.
- Avoid complex or bookish words.
- Use respectful, neutral, inclusive language.
- Avoid religious greetings or phrases.
- Prefer practical sentences that can be followed in the field.

Greeting Rule:
Use only: नमस्कार, प्रणाम, or किसान भाइयो बहनो

--------------------------------------------------

Formatting Rules:

- Output must be plain text only.
- No bullets, no symbols, no emojis.
- Everything must be speakable aloud.
- Be detailed but conversational.
""").strip()

# ---------------------------------------------------
# ✅ PROMPT BUILDERS
# ---------------------------------------------------
def build_raw_prompt():
    user_input_block = (
        f"Growth Stage: {INPUT_DATA['Growth Stage']}\n"
        f"Weather: {INPUT_DATA['Weather']}\n"
        f"Soil Type: {INPUT_DATA['Soil Type']}\n"
        f"Farming Practice: {INPUT_DATA['Farming Practice']}\n"
        f"Region: {INPUT_DATA['Region']}\n"
        f"Language: {INPUT_DATA['Language']}\n"
        f"Crop: {INPUT_DATA['Crop']}\n"
        f"Stress: {INPUT_DATA['Stress']}\n"
    )

    formatted_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{SYSTEM_PROMPT}\n\n"
        "### Input:\n"
        f"{user_input_block}\n\n"
        "### Response:\n"
    )

    return formatted_prompt


def build_primed_prompt():
    user_input_block = (
        f"Growth Stage: {INPUT_DATA['Growth Stage']}\n"
        f"Weather: {INPUT_DATA['Weather']}\n"
        f"Soil Type: {INPUT_DATA['Soil Type']}\n"
        f"Farming Practice: {INPUT_DATA['Farming Practice']}\n"
        f"Region: {INPUT_DATA['Region']}\n"
        f"Language: {INPUT_DATA['Language']}\n"
        f"Crop: {INPUT_DATA['Crop']}\n"
        f"Stress: {INPUT_DATA['Stress']}\n"
    )

    formatted_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{SYSTEM_PROMPT}\n\n"
        "### Input:\n"
        f"{user_input_block}\n\n"
        "### Response:\n"
        "<unused0>"   # Prime the model to start reasoning
    )

    return formatted_prompt

# ---------------------------------------------------
# ✅ INFERENCE
# ---------------------------------------------------
def run_inference():
    seed_everything(42)

    print("=" * 80)
    print("AGRI-EXPERT INFERENCE (HuggingFace Model - Updated Scenario)")
    print("=" * 80)
    print(f"\nModel: {MODEL_PATH}")

    # --- LOAD MODEL AND TOKENIZER ---
    print("\nLoading model and tokenizer...")
    start_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    # --- BUILD PROMPT ---
    formatted_prompt = build_primed_prompt()

    print("\n" + "=" * 80)
    print("INPUT DATA:")
    print("=" * 80)
    for key, value in INPUT_DATA.items():
        print(f"  {key}: {value}")

    # --- TOKENIZE ---
    input_ids = tokenizer.encode(
        formatted_prompt,
        add_special_tokens=True,
        return_tensors="pt"
    )
    print(f"\nInput token length: {input_ids.shape[1]}")
    input_ids = input_ids.to(model.device)
    input_length = input_ids.shape[1]

    # --- GENERATE ---
    print("\nGenerating response...")
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=3000,
            temperature=0.7,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generation_time = time.perf_counter() - t0

    # --- DECODE ---
    generated_tokens = outputs[0][input_length:]
    generated_token_count = len(generated_tokens)
    tokens_per_sec = generated_token_count / generation_time if generation_time > 0 else 0

    full_output = "<unused0>" + tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False
    )

    # --- DISPLAY RESULTS ---
    print("\n" + "=" * 80)
    print("FULL OUTPUT:")
    print("=" * 80)
    print(full_output)

    print("\n" + "=" * 80)
    print(f"Time: {generation_time:.2f}s | Tokens: {generated_token_count} | Tokens/sec: {tokens_per_sec:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    run_inference()
