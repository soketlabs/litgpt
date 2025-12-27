import sys
import torch
import time
from pathlib import Path
from litgpt.tokenizer import Tokenizer
from litgpt.utils import lazy_load
from litgpt.generate.base import generate
from litgpt.lora import GPT, Config

torch.set_float32_matmul_precision('high')

def run_inference():
    # --- PATHS ---
    base_dir = Path("/projects/data/teams/tts_team/agri_training/checkpoints/google/gemma-3-27b-it/google/gemma-3-27b-it")
    adapter_dir = Path("/projects/data/teams/tts_team/agri_training/test_16000_out_dir/step-000800")
    
    adapter_file = adapter_dir / "lit_model.pth.lora"

    print(f"--- Final Manual Inference ---")
    print(f"Base Model: {base_dir}")
    print(f"Adapter:    {adapter_file}")

    # --- STEP 1: LOAD & PATCH CONFIG ---
    print("Loading Base Configuration...")
    config = Config.from_file(base_dir / "model_config.yaml")
    print("Injecting LoRA parameters (Rank=16, Alpha=16)...")
    config.lora_r = 16
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.lora_query = True
    config.lora_key = True
    config.lora_value = True
    config.lora_projection = True
    config.lora_mlp = True
    config.lora_head = True

    # --- STEP 2: BUILD MODEL ---
    print("Building LoRA-enabled GPT model...")
    with torch.device("meta"):
        model = GPT(config)
    model = model.to(dtype=torch.bfloat16)
    model = model.to_empty(device="cuda")

    # --- STEP 3: LOAD WEIGHTS ---
    print("Loading Base Model weights...")
    base_checkpoint = lazy_load(base_dir / "lit_model.pth")
    model.load_state_dict(base_checkpoint, strict=False)

    print("Loading Adapter weights...")
    adapter_checkpoint = lazy_load(adapter_file)
    adapter_checkpoint = adapter_checkpoint.get("model", adapter_checkpoint)
    keys = model.load_state_dict(adapter_checkpoint, strict=False)

    model.eval()

    print(f"Adapter Load Status: {keys}")

    model.eval()
    model.max_seq_length = 4096
    model.set_kv_cache(batch_size=1)
    device = torch.device("cuda")

    model.cuda()

    if hasattr(model, 'mask_cache') and model.mask_cache is not None:
        model.mask_cache = model.mask_cache.to("cuda")

    if hasattr(model.transformer, 'h'):
        for block in model.transformer.h:
            if hasattr(block.attn, 'kv_cache'):
                kvc = block.attn.kv_cache
                if hasattr(kvc, 'k') and kvc.k is not None:
                    kvc.k = kvc.k.to(device)
                if hasattr(kvc, 'v') and kvc.v is not None:
                    kvc.v = kvc.v.to(device)
                if hasattr(block.attn, 'adapter_kv_cache') and block.attn.adapter_kv_cache is not None:
                     k, v = block.attn.adapter_kv_cache
                     block.attn.adapter_kv_cache = (k.to(device), v.to(device))

    # --- STEP 4: TOKENIZER ---
    print("Loading Tokenizer...")
    tokenizer = Tokenizer(base_dir)

    # --- STEP 5: GENERATE ---
    raw_prompt = "मेरी गेहूँ की फसल में पत्तियाँ पीली हो रही हैं, इसका कारण क्या हो सकता है और इसे ठीक करने के लिए मुझे क्या करना चाहिए"
    
    formatted_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{raw_prompt}\n\n"
        "### Response:\n"
    )
    
    print(f"\n--- formatted Prompt ---\n{formatted_prompt}\n")
    
    encoded = tokenizer.encode(formatted_prompt, device="cuda")
    
    t0 = time.perf_counter()
    y = generate(
        model, 
        encoded, 
        max_returned_tokens=4096, 
        temperature=0.7, 
        top_k=50,
        eos_id=tokenizer.eos_id
    )
    t = time.perf_counter() - t0

    generated_tokens = y[encoded.size(0):] 
    output = tokenizer.decode(generated_tokens).strip()
    
    print("\n" + "="*40 + "\nOUTPUT:\n" + "="*40)
    print(output)
    
    print("\n" + "="*40)
    print("OUTPUT:")
    print("="*40)
    print(output)
    print("="*40)
    print(f"Time: {t:.2f}s")

if __name__ == "__main__":
    run_inference()
