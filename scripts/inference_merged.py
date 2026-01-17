import sys
import torch
import time
import re
import random
import textwrap
import numpy as np
from pathlib import Path
from litgpt.tokenizer import Tokenizer
from litgpt.utils import lazy_load
from litgpt.generate.base import generate
from litgpt.model import GPT, Config

torch.set_float32_matmul_precision('high')

# --- DETERMINISM SETUP ---
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- INPUT DATA ---
INPUT_VARIANTS = [
    {
        "Growth Stage": "कल्ले निकलना",
        "Weather": "नमी अधिक और बादल छाए हुए",
        "Soil Type": "दोमट मिट्टी",
        "Farming Practice": "सामान्य खेती",
        "Region": "बिहार",
        "Language": "Hindi",
        "Crop": "धान",
        "Stress": "भूरा तना फुदका",
    },
    {
        "Growth Stage": "बाल निकलना",
        "Weather": "ठंडी रात और हल्की नमी",
        "Soil Type": "दोमट मिट्टी",
        "Farming Practice": "सामान्य खेती",
        "Region": "उत्तर प्रदेश",
        "Language": "Hindi",
        "Crop": "गेहूं",
        "Stress": "पीला रतुआ रोग",
    },
    {
        "Growth Stage": "फूल आना",
        "Weather": "गरम और आर्द्र मौसम",
        "Soil Type": "काली मिट्टी",
        "Farming Practice": "सामान्य खेती",
        "Region": "महाराष्ट्र",
        "Language": "Hindi",
        "Crop": "कपास",
        "Stress": "सफेद मक्खी",
    },
]

INPUT_DATA = random.choice(INPUT_VARIANTS)


def run_inference():
    # SET FIXED SEED
    seed_everything(9)

    # --- PATH TO MERGED CHECKPOINT ---
    checkpoint_dir = Path("/projects/data/teams/tts_team/agri_training/finetuned_checkpoints_parquet/gemma3-27b-it-lora-2700/step-005625/lora_2700_2ndCrash/step-007750/")

    print(f"--- Agri-Expert Inference (Merged Checkpoint) ---")
    
    # --- STEP 1: CONFIG ---
    print("Loading Config...")
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    # --- STEP 2: BUILD MODEL ---
    print("Building Model...")
    with torch.device("meta"):
        model = GPT(config)
    model = model.to(dtype=torch.bfloat16).to_empty(device="cuda")

    # --- STEP 3: LOAD WEIGHTS ---
    print("Loading Weights...")
    checkpoint = lazy_load(checkpoint_dir / "lit_model.pth")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # --- KV CACHE ---
    model.max_seq_length = 8192 
    model.set_kv_cache(batch_size=1)
    
    # Force GPU moves
    device = torch.device("cuda")
    if hasattr(model, 'mask_cache') and model.mask_cache is not None:
        model.mask_cache = model.mask_cache.to(device)
    if hasattr(model.transformer, 'h'):
        for block in model.transformer.h:
            if hasattr(block.attn, 'kv_cache'):
                kvc = block.attn.kv_cache
                if hasattr(kvc, 'k'): kvc.k = kvc.k.to(device)
                if hasattr(kvc, 'v'): kvc.v = kvc.v.to(device)

    # --- STEP 4: TOKENIZER ---
    tokenizer = Tokenizer(checkpoint_dir)

    # --- STEP 5: PROMPT WITH FORCED REASONING ---
    system_prompt = textwrap.dedent("""
    Cognitive Discipline Protocol (Highest Priority):

    Before producing any advisory, you must perform explicit internal reasoning.

    Output Order is Mandatory and Strict:
    1. The very first output token must be exactly <unused0>
    2. Immediately after <unused0>, produce detailed step by step reasoning in English strictly covering all analytical checks required by this system prompt
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
    f"{system_prompt}\n\n"
    "### Input:\n"
    f"{user_input_block}\n\n"
    "### Response:\n"
    )
    
    print(f"\n--- Sending Prompt ---\n{formatted_prompt}\n")
    
    # Tokenize
    encoded = tokenizer.encode(formatted_prompt, device="cuda")
    
    # --- GENERATION ---
    t0 = time.perf_counter()
    y = generate(
        model, 
        encoded, 
        max_returned_tokens=encoded.size(0) + 3000, 
        temperature=0.7, 
        top_k=50,
        eos_id=tokenizer.eos_id
    )
    t = time.perf_counter() - t0

    # Tokens and throughput
    generated_token_count = y.numel() - encoded.numel()
    tokens_per_sec = generated_token_count / t if t > 0 else 0.0

    # --- DECODING AND SPLITTING ---
    generated_tokens = y[encoded.size(0):] 
    full_output = tokenizer.decode(generated_tokens).strip()

    print("\n" + "="*40 + "\nFULL OUTPUT:\n" + "="*40)
    print(full_output)

    print("="*40)
    print(f"Time: {t:.2f}s | Tokens: {generated_token_count} | Tokens/sec: {tokens_per_sec:.2f}")

if __name__ == "__main__":
    run_inference()