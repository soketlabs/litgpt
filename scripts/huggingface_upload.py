
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
BASE_MODEL_ID = "google/gemma-3-27b-it"
TARGET_MODEL_ID = "iota-10/saarthi-v1"
LOCAL_CHECKPOINT_PATH = '/projects/data/teams/tts_team/agri_training/finetuned_checkpoints_parquet/gemma3-27b-it-lora-2700/step-005625/lora_2700_2ndCrash/step-007750/hf_compatible/model.pth'

def main():
    print(f"Loading base model: {BASE_MODEL_ID}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 2. Load Base Model (with default tied weights)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("Base model loaded.")

    # 3. Untie weights explicitly before changing config or updating state dict
    print("Checking for weight tying...")
    if model.lm_head.weight is model.model.language_model.embed_tokens.weight:
        print("Weights are tied. Breaking the tie now...")
        # Create a new parameter with a copy of the weights to physically separate the memory
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
        print("Weights untied successfully.")
    else:
        print("Weights were already untied.")

    # 4. Update Config
    print("Updating configuration to tie_word_embeddings = False")
    model.config.text_config.tie_word_embeddings = False
    
    # 5. Load Local Checkpoint
    print(f"Loading local checkpoint from: {LOCAL_CHECKPOINT_PATH}")
    if not os.path.exists(LOCAL_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {LOCAL_CHECKPOINT_PATH}")
        
    checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location="cpu")  # Load to CPU first to avoid OOM
    
    # 6. Separate Checkpoint into Components
    print("Separating checkpoint weights...")
    rm_params = {}
    lm_head_weights = None
    
    for k, v in checkpoint.items():
        if k.startswith('model.'):
            # Strip 'model.' prefix for language model loading
            rm_params[k[6:]] = v
        elif k.startswith('lm_head.'):
            # Prepare lm_head weights
            lm_head_weights = v
        else:
            rm_params[k] = v
            
    if lm_head_weights is None:
        raise ValueError("Could not find lm_head weights in the checkpoint!")
        
    print(f"Found {len(rm_params)} body parameters and lm_head weights.")

    # 7. Load State Dicts
    print("Loading body weights...")
    
    missing, unexpected = model.language_model.load_state_dict(rm_params, strict=True)
    if len(missing) > 0:
        print(f"Missing keys in body: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected keys in body: {unexpected}")
    print("Body weights loaded.")

    print("Loading lm_head weights...")
    model.lm_head.load_state_dict({'weight': lm_head_weights}, strict=True)
    print("lm_head weights loaded.")
    
    # ---------------------------------------------------------
    # VERIFICATION: Run Inference Locally
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("VERIFICATION: Running local inference...")
    print("="*50)
    
    from transformers import TextStreamer
    
    messages = [
        {"role": "system", "content": "You are a helpful agronomy expert providing crop advisory to farmers based on location and weather conditions."},
        {"role": "user", "content": "I want crop advisory for varanasi. Its jan and I am thinking of sowing sugercase. Rain has been very fragmented. Think hard and give me advisory in hindi"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print("Generating response...\n")
    model.eval()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.7,
            streamer=streamer
        )
            
    print("\n" + "="*50)
    user_input = input("\nDoes the output look correct? Proceed to upload? (y/n): ")
    
    if user_input.lower() != 'y':
        print("Upload cancelled by user.")
        return

    # 8. Push to Hub
    print(f"Pushing to Hub: {TARGET_MODEL_ID}")
    
    # Push tokenizer first
    tokenizer.push_to_hub(TARGET_MODEL_ID)
    
    # Push model
    model.push_to_hub(TARGET_MODEL_ID)
    
    print("Upload complete!")

if __name__ == "__main__":
    main()
