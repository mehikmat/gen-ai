import tiktoken
import torch
from importlib.metadata import version

# import utilities
from supplementary import (
    GPTModel,
    create_dataloader_v1,
    train_model_simple,
    plot_losses
)

# print version of libraries
pkgs = ["tiktoken",
        "torch",
        'matplotlib'
        ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

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

# [1] read raw text data
with open("/data/the-verdict.txt", 'r', encoding="utf-8") as f:
    text_data = f.read()

# create GPT BPE tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Total Characters:", total_characters)
print("Total Tokens:", total_tokens)

# [2] split input data into training and valuation datasets
train_ratio = 0.90  # Train/validation ratio
split_idx = int(train_ratio * total_characters)
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("\nTrain loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("\nTraining tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

# [3] init empty GPT model
# seed for random number generator used in various stuff like initializing weights, shuffling data, etc.
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# [4] train the model
num_epochs = 10  # number of iterations, one iteration includes whole data processing
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# [5] save trained model
torch.save(model.state_dict(), "/Users/mehikmat/proj/gen-ai/model/gpt_llm_model.pth")
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
