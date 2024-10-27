from importlib.metadata import version

import tiktoken
import torch

# Relative import from the gpt_download.py contained in this folder
from gpt_download import download_and_load_gpt2
from supplementary import GPTModel, load_weights_into_gpt
from supplementary import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)

# print version of libraries
pkgs = ["tiktoken",
        "torch",
        'tensorflow',
        'tqdm'
        ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
print("Settings:", settings)
print("Parameter dictionary keys:", params.keys())

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-key-value bias
}

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()

load_weights_into_gpt(gpt, params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)

# using model
tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)

token_ids = generate_text_simple(
    model=gpt,
    idx=text_to_token_ids("Correct spelling: I am going to eaten", tokenizer).to(device),
    max_new_tokens=20,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
