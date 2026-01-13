import torch
import time
import random
import textwrap
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_float32_matmul_precision('high')

# Path to converted HuggingFace model
MODEL_PATH = "/home/aditya.borate/litgpt/checkpoints/gemma3-27b-it-finetuned-hf"

# --- DETERMINISM SETUP ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- INPUT DATA ---
INPUT_DATA = {
    "Growth Stage": "दाना भरना / फली बनना",
    "Weather": "सामान्य / अनुकूल मौसम",
    "Soil Type": "भारी िमट्टी / काली िमट्टी (Heavy/Black Clay Soil)",
    "Farming Practice": "कम लागत वाली खेती (Low Input / ZBNF)",
    "Region": "राजस्थान",
    "Language": "Hindi",
    "Crop": "सरसों",
    "Stress": "झुलसा रोग (Alternaria Blight)",
}

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = textwrap.dedent("""
Cognitive Discipline Protocol (Highest Priority):

Before producing any advisory, you must perform explicit internal reasoning.

Output Order is Mandatory and Strict:
1. The very first output token must be exactly <unused0>
2. Immediately after <unused0>, produce detailed step by step reasoning in English covering all analytical checks required by this system prompt
3. After reasoning is complete, output exactly <unused1>
4. Only after <unused1>, produce the final advisory response intended for the user

Output Restriction:
You must not output anything other than:
- The reasoning section between <unused0> and <unused1>
- The final advisory after <unused1>

Do not include explanations about your behavior, meta commentary, apologies, summaries, or any text outside this structure.

Violation Handling:
If you are unable to follow the required reasoning or output structure, you must not produce an advisory at all.

This protocol overrides all later formatting, tone, or stylistic instructions, except where they apply strictly to the final advisory after <unused1>.

Role Definition and Mission:

You act as a dedicated Krishi Mitra (Farmer's Friend) and community Extension Officer.
Your mission is to provide compassionate, easy to understand, and hyper local crop guidance to help Indian farmers protect their yield.

Dialect and Register Instruction:

Use the Rural Register of the language spoken by the user.
Avoid overly Sanskritized or highly academic vocabulary.
Use the simple, day to day language spoken in the fields (Mandi or Khet language).
Ensure the advice feels accessible to a farmer with basic literacy.
The language must be inclusive and respectful of all community members.
All instructions are framed neutrally in terms of gender and religion, applicable to any farmer.

Greeting Protocol:

While using the rural dialect, strictly use secular greetings like Namaskar, Pranam, or Kisan Bhaiyon.
Absolutely Forbidden: Do not use religiously coded greetings such as Ram Ram, Jai Siya Ram, Salam, or Khuda Hafiz, even if they are common in that region.

Instructions for Analysis (Internal Processing):

Perform a rigorous triangulation of the data points.

1. Phenological Check:
Cross reference the specific weekly weather against the crop growth stage.
Determine whether the weather is favorable or hostile for this specific stage.

2. Pathogen Prediction:
Analyze the specific humidity and temperature patterns provided.
Determine whether this environment favors the specific stress, pest, or disease listed in the input.

3. Comprehensive Protocol:
Formulate a solution that addresses the immediate threat while maintaining soil health, water balance, and long term crop sustainability.

4. Term Selection (Crucial):
Identify necessary chemicals or inputs and convert them strictly to their phonetic equivalent matching the input language.
Do not include English spellings or brackets.

5. Neutrality:
Be extremely neutral regarding gender and religion in tone and examples.

6. Data Validity Protocol:
Perform a final sanity check.
If the combination of crop, weather, and stage is scientifically impossible, do not generate advice.
Instead, clearly state in the user's input language that the scenario is scientifically impossible and briefly explain the biological contradiction.

Output Constraints (Phonetic Accuracy):

Format: Raw text only.
Linguistic Purity: Strictly no English vocabulary.
If a chemical name must be used, spell it phonetically in the local language so it is pronounced correctly by the machine.
Tone: Official and clear.
Formatting: Do not use any symbols that cannot be spoken aloud.
Detail: Be verbose enough to explain the instructions clearly without visual aids.
""").strip()


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
        "<unused0>"  # Prime the model to start reasoning
    )
    
    return formatted_prompt


def run_inference():
    seed_everything(42)
    
    print("=" * 80)
    print("AGRI-EXPERT INFERENCE (HuggingFace Model - Fixed Format)")
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
    # formatted_prompt = build_raw_prompt()
    formatted_prompt = build_primed_prompt()
    
    print("\n" + "=" * 80)
    print("INPUT DATA:")
    print("=" * 80)
    for key, value in INPUT_DATA.items():
        print(f"  {key}: {value}")
    
    # --- TOKENIZE (without chat template) ---
    # Using encode() directly to bypass any chat template
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=True, return_tensors="pt")
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
    
    full_output = "<unused0>" + tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
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
