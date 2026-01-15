import gc
import json
import shutil
import sys
import torch
from pathlib import Path
from typing import Dict

sys.path.insert(0, "/home/aditya.borate/litgpt")

from safetensors.torch import save_file
from safetensors import safe_open

# ============= CONFIGURATION =============
LITGPT_LORA_CHECKPOINT = Path("/projects/data/teams/tts_team/agri_training/finetuned_checkpoints_parquet/gemma3-27b-it-lora-2700/step-005625/lora_2700_2ndCrash/step-008375/")
BASE_MODEL_DIR = Path("/projects/data/teams/tts_team/agri_training/checkpoints/google/gemma-3-27b-it/google/gemma-3-27b-it")
HF_REFERENCE_MODEL = Path("/home/aditya.borate/litgpt/Gemma-3-27b-it")
OUTPUT_DIR = Path("/home/aditya.borate/litgpt/checkpoints/step-008375-correct-merged")

# LitGPT config
LITGPT_VOCAB_SIZE = 262144
HF_VOCAB_SIZE = 262208
HIDDEN_SIZE = 5376
NUM_LAYERS = 62
NUM_HEADS = 32
NUM_KV_HEADS = 16
HEAD_SIZE = 128

# ============= STEP 1: MERGE LORA USING LITGPT =============
def merge_lora_with_litgpt():
    """Use LitGPT's official merge_lora function."""
    print("=" * 80)
    print("STEP 1: MERGING LORA WEIGHTS USING LITGPT")
    print("=" * 80)
    
    merged_path = LITGPT_LORA_CHECKPOINT / "lit_model.pth"
    if merged_path.exists():
        print(f"Merged model already exists at {merged_path}")
        return merged_path
    
    from litgpt.scripts.merge_lora import merge_lora
    
    print(f"Merging LoRA weights from: {LITGPT_LORA_CHECKPOINT}")
    
    # Call LitGPT's merge function
    merge_lora(
        checkpoint_dir=LITGPT_LORA_CHECKPOINT,
        pretrained_checkpoint_dir=BASE_MODEL_DIR,
        precision="bf16-true"
    )
    
    print(f"Merged model saved to: {merged_path}")
    return merged_path


# ============= STEP 2: CONVERT TO HF FORMAT =============
def split_qkv(qkv_weight, n_head, n_kv_head, head_size):
    """Split QKV fused weight into separate Q, K, V weights."""
    q_size = n_head * head_size
    kv_size = n_kv_head * head_size
    
    q, k, v = qkv_weight.split([q_size, kv_size, kv_size], dim=0)
    return q, k, v


def pad_embedding(weight, target_vocab_size):
    """Pad embedding/lm_head weights from LitGPT vocab size to HF vocab size."""
    current_vocab, hidden = weight.shape
    if current_vocab >= target_vocab_size:
        return weight
    
    padding = torch.zeros((target_vocab_size - current_vocab, hidden), dtype=weight.dtype)
    return torch.cat([weight, padding], dim=0)


def convert_litgpt_to_hf(merged_model_path: Path):
    """Convert merged LitGPT weights to HuggingFace format."""
    print("\n" + "=" * 80)
    print("STEP 2: CONVERTING TO HUGGINGFACE FORMAT")
    print("=" * 80)
    
    print(f"Loading merged LitGPT model from: {merged_model_path}")
    lit_sd = torch.load(merged_model_path, map_location='cpu')
    
    hf_sd = {}
    
    print("\nConverting weights...")
    
    # 1. Embeddings - pad vocab size
    embed_weight = lit_sd["transformer.wte.weight"]
    padded_embed = pad_embedding(embed_weight, HF_VOCAB_SIZE)
    hf_sd["language_model.model.embed_tokens.weight"] = padded_embed
    print(f"  embed_tokens: {list(embed_weight.shape)} -> {list(padded_embed.shape)}")
    
    # 2. Final layer norm
    hf_sd["language_model.model.norm.weight"] = lit_sd["transformer.ln_f.weight"]
    print(f"  norm.weight: {list(lit_sd['transformer.ln_f.weight'].shape)}")
    
    # 3. Process each transformer layer
    for layer_idx in range(NUM_LAYERS):
        lit_prefix = f"transformer.h.{layer_idx}"
        hf_prefix = f"language_model.model.layers.{layer_idx}"
        
        # Input layernorm
        hf_sd[f"{hf_prefix}.input_layernorm.weight"] = lit_sd[f"{lit_prefix}.norm_1.weight"]
        
        # QKV projection - split into separate Q, K, V
        qkv_weight = lit_sd[f"{lit_prefix}.attn.qkv.weight"]
        q, k, v = split_qkv(qkv_weight, NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE)
        hf_sd[f"{hf_prefix}.self_attn.q_proj.weight"] = q
        hf_sd[f"{hf_prefix}.self_attn.k_proj.weight"] = k
        hf_sd[f"{hf_prefix}.self_attn.v_proj.weight"] = v
        
        # O projection
        hf_sd[f"{hf_prefix}.self_attn.o_proj.weight"] = lit_sd[f"{lit_prefix}.attn.proj.weight"]
        
        # Q/K norms (Gemma 3 specific)
        if f"{lit_prefix}.attn.norm_q.weight" in lit_sd:
            hf_sd[f"{hf_prefix}.self_attn.q_norm.weight"] = lit_sd[f"{lit_prefix}.attn.norm_q.weight"]
        if f"{lit_prefix}.attn.norm_k.weight" in lit_sd:
            hf_sd[f"{hf_prefix}.self_attn.k_norm.weight"] = lit_sd[f"{lit_prefix}.attn.norm_k.weight"]
        
        # Post-attention layernorm
        if f"{lit_prefix}.post_attention_norm.weight" in lit_sd:
            hf_sd[f"{hf_prefix}.post_attention_layernorm.weight"] = lit_sd[f"{lit_prefix}.post_attention_norm.weight"]
        
        # Pre-feedforward layernorm  
        if f"{lit_prefix}.norm_2.weight" in lit_sd:
            hf_sd[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = lit_sd[f"{lit_prefix}.norm_2.weight"]
        
        # MLP weights
        hf_sd[f"{hf_prefix}.mlp.gate_proj.weight"] = lit_sd[f"{lit_prefix}.mlp.fc_1.weight"]
        hf_sd[f"{hf_prefix}.mlp.up_proj.weight"] = lit_sd[f"{lit_prefix}.mlp.fc_2.weight"]
        hf_sd[f"{hf_prefix}.mlp.down_proj.weight"] = lit_sd[f"{lit_prefix}.mlp.proj.weight"]
        
        # Post-MLP layernorm
        if f"{lit_prefix}.post_mlp_norm.weight" in lit_sd:
            hf_sd[f"{hf_prefix}.post_feedforward_layernorm.weight"] = lit_sd[f"{lit_prefix}.post_mlp_norm.weight"]
        
        if (layer_idx + 1) % 10 == 0:
            print(f"  Converted layer {layer_idx + 1}/{NUM_LAYERS}")
    
    print(f"\nConverted {len(hf_sd)} language model weights")
    
    del lit_sd
    gc.collect()
    
    return hf_sd


# ============= STEP 3: COPY VISION TOWER =============
def copy_vision_tower():
    """Copy vision tower and multi-modal projector from original HF model."""
    print("\n" + "=" * 80)
    print("STEP 3: COPYING VISION TOWER FROM ORIGINAL HF MODEL")
    print("=" * 80)
    
    vision_weights = {}
    
    # Load vision tower weights from original safetensors
    index_path = HF_REFERENCE_MODEL / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find which shards contain vision and multi-modal weights
    vision_keys = [k for k in weight_map.keys() if k.startswith(("vision_tower.", "multi_modal_projector."))]
    print(f"Found {len(vision_keys)} vision-related weights to copy")
    
    # Group by shard file
    shard_keys = {}
    for key in vision_keys:
        shard = weight_map[key]
        if shard not in shard_keys:
            shard_keys[shard] = []
        shard_keys[shard].append(key)
    
    # Load from each shard
    for shard, keys in shard_keys.items():
        shard_path = HF_REFERENCE_MODEL / shard
        print(f"  Loading from {shard}...")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                vision_weights[key] = f.get_tensor(key)
    
    print(f"Copied {len(vision_weights)} vision tower weights")
    return vision_weights


# ============= STEP 4: CREATE CONFIG & SAVE =============
def create_hf_config():
    """Create HuggingFace config.json."""
    config = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "boi_token_index": 255999,
        "eoi_token_index": 256000,
        "eos_token_id": [1, 106],
        "image_token_index": 262144,
        "initializer_range": 0.02,
        "mm_tokens_per_image": 256,
        "model_type": "gemma3",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_logit_softcapping": None,
            "cache_implementation": "hybrid",
            "final_logit_softcapping": None,
            "head_dim": HEAD_SIZE,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": HIDDEN_SIZE,
            "initializer_range": 0.02,
            "intermediate_size": 21504,
            "max_position_embeddings": 131072,
            "model_type": "gemma3_text",
            "num_attention_heads": NUM_HEADS,
            "num_hidden_layers": NUM_LAYERS,
            "num_key_value_heads": NUM_KV_HEADS,
            "query_pre_attn_scalar": 168,
            "rms_norm_eps": 1e-06,
            "rope_local_base_freq": 10000.0,
            "rope_scaling": {
                "factor": 8.0,
                "rope_type": "linear"
            },
            "rope_theta": 1000000.0,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "tie_word_embeddings": True,
            "use_cache": True,
            "vocab_size": HF_VOCAB_SIZE
        },
        "torch_dtype": "bfloat16",
        "transformers_version": "4.50.0",
        "vision_config": {
            "hidden_size": 1152,
            "image_size": 896,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "patch_size": 14,
            "vision_use_head": False
        }
    }
    return config


def save_as_safetensors(all_weights: Dict[str, torch.Tensor], output_dir: Path):
    """Save weights as sharded safetensors files with index."""
    print("\n" + "=" * 80)
    print("STEP 4: SAVING AS SAFETENSORS")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    MAX_SHARD_SIZE = 5 * 1024 * 1024 * 1024  # 5GB per shard
    
    current_shard = {}
    current_size = 0
    shard_idx = 1
    weight_map = {}
    
    sorted_keys = sorted(all_weights.keys())
    
    for key in sorted_keys:
        tensor = all_weights[key]
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_size + tensor_size > MAX_SHARD_SIZE and current_shard:
            # Save current shard
            shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(current_shard, output_dir / shard_name)
            print(f"  Saved {shard_name} ({len(current_shard)} tensors)")
            
            for k in current_shard:
                weight_map[k] = shard_name
            
            current_shard = {}
            current_size = 0
            shard_idx += 1
        
        current_shard[key] = tensor
        current_size += tensor_size
    
    # Save final shard
    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(current_shard, output_dir / shard_name)
        print(f"  Saved {shard_name} ({len(current_shard)} tensors)")
        for k in current_shard:
            weight_map[k] = shard_name
    
    total_shards = shard_idx
    
    # Rename files with correct total
    for i in range(1, total_shards + 1):
        old_name = output_dir / f"model-{i:05d}-of-XXXXX.safetensors"
        new_name = output_dir / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        old_name.rename(new_name)
        
        # Update weight_map
        for k, v in list(weight_map.items()):
            if v == f"model-{i:05d}-of-XXXXX.safetensors":
                weight_map[k] = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
    
    # Create index file
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in all_weights.values())},
        "weight_map": weight_map
    }
    
    with open(output_dir / "model.safetensors.index.json", 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"\nSaved {total_shards} shards with {len(weight_map)} total weights")


def copy_tokenizer_files(output_dir: Path):
    """Copy tokenizer files from original HF model."""
    print("\n" + "=" * 80)
    print("STEP 5: COPYING TOKENIZER FILES")
    print("=" * 80)
    
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model", 
        "tokenizer_config.json"
    ]
    
    for fname in tokenizer_files:
        src = HF_REFERENCE_MODEL / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
            print(f"  Copied {fname}")
        else:
            print(f"  Warning: {fname} not found")
    
    # Also copy other config files from HF model
    other_files = ["preprocessor_config.json", "processor_config.json", "special_tokens_map.json", "chat_template.json", "added_tokens.json", "generation_config.json"]
    for fname in other_files:
        src = HF_REFERENCE_MODEL / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
            print(f"  Copied {fname}")


def main():
    print("=" * 80)
    print("LITGPT TO HUGGINGFACE CONVERSION FOR GEMMA-3-27B-IT")
    print("=" * 80)
    print(f"\nInput LitGPT checkpoint: {LITGPT_LORA_CHECKPOINT}")
    print(f"HF reference model: {HF_REFERENCE_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Merge LoRA using LitGPT
    merged_path = merge_lora_with_litgpt()
    
    # Step 2: Convert to HF format
    hf_language_weights = convert_litgpt_to_hf(merged_path)
    
    # Step 3: Copy vision tower
    vision_weights = copy_vision_tower()
    
    # Combine all weights
    all_weights = {**hf_language_weights, **vision_weights}
    print(f"\nTotal weights: {len(all_weights)}")
    
    # Step 4: Save as safetensors
    save_as_safetensors(all_weights, OUTPUT_DIR)
    
    # Save config
    config = create_hf_config()
    with open(OUTPUT_DIR / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config.json")
    
    # Step 5: Copy tokenizer files
    copy_tokenizer_files(OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
