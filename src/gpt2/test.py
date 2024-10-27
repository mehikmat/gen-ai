import tiktoken
import torch

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("Hello world"))
print(torch.tensor(tokenizer.encode("hello world")))
