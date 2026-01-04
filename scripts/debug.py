import torch
import sys
from pathlib import Path

# Add litgpt to path
sys.path.insert(0, str(Path.home() / "litgpt"))

from litgpt.model import GPT, Config

# Load your config
config = Config.from_file(
    "/projects/data/teams/tts_team/agri_training/test_16000_out_dir_8_1_three_combined_dataset_gemma/step-004800/model_config.yaml"
)

print("Config loaded:")
print(f"n_head: {config.n_head}")
print(f"n_query_groups: {config.n_query_groups}")
print(f"head_size: {config.head_size}")
print(f"block_size: {config.block_size}")
print(f"sliding_window_size: {config.sliding_window_size}")

# Try to instantiate model
with torch.device("meta"):
    model = GPT(config)

print("\nModel created successfully")

# Check a specific layer's KV cache setup
layer_0 = model.transformer.h[0]
print(f"\nLayer 0 attention type: {type(layer_0.attn)}")
print(f"Has kv_cache: {hasattr(layer_0.attn, 'kv_cache')}")

if hasattr(layer_0.attn, 'kv_cache'):
    kv_cache = layer_0.attn.kv_cache
    print(f"KV cache type: {type(kv_cache)}")
    print(f"KV cache k shape: {kv_cache.k.shape if hasattr(kv_cache, 'k') else 'N/A'}")
    print(f"KV cache v shape: {kv_cache.v.shape if hasattr(kv_cache, 'v') else 'N/A'}")