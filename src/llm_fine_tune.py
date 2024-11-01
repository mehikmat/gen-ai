from importlib.metadata import version

from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig

# print version of libraries
pkgs = ["transformers",
        "torch",
        "datasets",
        "bitsandbytes",
        "peft"
        ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

model_name = "microsoft/phi-2"

# Configuration to load model in 4-bit quantized
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type='nf4',
                                bnb_4bit_compute_dtype='float16',
                                bnb_4bit_use_double_quant=True)
# Loading model with compatible settings
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map='auto',
#                                              quantization_config=bnb_config,
#                                              trust_remote_code=True)
# Setting up tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          add_eos_token=True,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

# load data
train_dataset = load_dataset('json',
                             data_dir='/Users/mehikmat/proj/gen-ai/data/phi-2',
                             split="train[0:1000]")
test_dataset = load_dataset('json',
                            data_dir='/Users/mehikmat/proj/gen-ai/data/phi-2',
                            split="train[1000:1100]")

print(train_dataset)
print(test_dataset)
