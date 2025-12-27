import sys
import torch
import time
from pathlib import Path
from litgpt.tokenizer import Tokenizer
from litgpt.utils import lazy_load
from litgpt.generate.base import generate
from litgpt.lora import GPT, Config

torch.set_float32_matmul_precision('high')

BASE_DIR = Path("/projects/data/teams/tts_team/agri_training/checkpoints/google/gemma-3-27b-it/google/gemma-3-27b-it")
ADAPTER_DIR = Path("/projects/data/teams/tts_team/agri_training/test_16000_out_dir/step-000800")

def load_model():
    print("--- Loading LoRA Model ---")
    config = Config.from_file(BASE_DIR / "model_config.yaml")
    
    config.lora_r = 16
    config.lora_alpha = 16
    config.lora_dropout = 0.1
    config.lora_query = True
    config.lora_key = True
    config.lora_value = True
    config.lora_projection = True
    config.lora_mlp = True
    config.lora_head = True

    with torch.device("meta"):
        model = GPT(config)
    
    model = model.to(dtype=torch.bfloat16)
    model = model.to_empty(device="cuda") 
    
    base_ckpt = lazy_load(BASE_DIR / "lit_model.pth")
    model.load_state_dict(base_ckpt, strict=False)
    
    adapter_file = ADAPTER_DIR / "lit_model.pth.lora"
    
    adapter_ckpt = lazy_load(adapter_file)
    adapter_ckpt = adapter_ckpt.get("model", adapter_ckpt)
    model.load_state_dict(adapter_ckpt, strict=False)
    
    model.eval()
    
    # Setup Cache
    model.max_seq_length = 8192  
    model.set_kv_cache(batch_size=1)
    
    # Force GPU
    device = torch.device("cuda")
    if hasattr(model, 'mask_cache') and model.mask_cache is not None:
        model.mask_cache = model.mask_cache.to(device)
    if hasattr(model.transformer, 'h'):
        for block in model.transformer.h:
            if hasattr(block.attn, 'kv_cache'):
                kvc = block.attn.kv_cache
                if hasattr(kvc, 'k'): kvc.k = kvc.k.to(device)
                if hasattr(kvc, 'v'): kvc.v = kvc.v.to(device)
            elif hasattr(block.attn, 'adapter_kv_cache'):
                c = block.attn.adapter_kv_cache
                if isinstance(c, tuple):
                    block.attn.adapter_kv_cache = (c[0].to(device), c[1].to(device))
                    
    return model

def build_prompt(history, new_question):
    """
    Constructs the full conversation string in Alpaca format.
    """
    full_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    
    # Add History
    for turn in history:
        full_prompt += f"### Instruction:\n{turn['user']}\n\n"
        full_prompt += f"### Response:\n{turn['bot']}\n\n"
        
    # Add New Question
    full_prompt += f"### Instruction:\n{new_question}\n\n"
    full_prompt += "### Response:\n"
    
    return full_prompt

def run_chat_simulation():
    model = load_model()
    tokenizer = Tokenizer(BASE_DIR)
        
    # --- CONVERSATION SIMULATION ---
    questions = [
        "What are the most effective ways to conserve water in rice farming?",
        "Can you explain the second method you mentioned in more detail?"  
    ]
    
    history = []

    for i, question in enumerate(questions):
        print(f"🔹 TURN {i+1} User: {question}")
        
        # 1. Build the full transcript
        prompt_text = build_prompt(history, question)
        
        # 2. Tokenize
        encoded = tokenizer.encode(prompt_text, device="cuda")
        input_length = encoded.size(0)  
        
        # 3. Generate
        t0 = time.perf_counter()
        y = generate(
            model, 
            encoded, 
            max_returned_tokens=input_length + 1000, 
            temperature=0.7, 
            top_k=50,
            eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0
     
        generated_tokens = y[input_length:]
        generated_part = tokenizer.decode(generated_tokens).strip()
        
        print(f"TURN {i+1} Model ({t:.2f}s):")
        print(generated_part)
        print("-" * 40)
        
        # 4. Update History
        history.append({"user": question, "bot": generated_part})

if __name__ == "__main__":
    run_chat_simulation()