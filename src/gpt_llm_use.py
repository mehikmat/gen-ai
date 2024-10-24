import tiktoken
import torch

# import utilities
from supplementary import (
    GPTModel,
    generate_text_simple,
    token_ids_to_text,
    text_to_token_ids
)

# LLM configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024 for 124M parameters)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-key-value bias
}

# load pre trained model
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("/Users/mehikmat/proj/gen-ai-rnn/model/gpt_llm_model.pth", map_location=device))

# prompt
start_context = "I found the couple at tea beneath"

tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ask model by giving the prompt
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer).to(device),
    max_new_tokens=2,
    context_size=GPT_CONFIG_124M["context_length"]
)

# model output
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
