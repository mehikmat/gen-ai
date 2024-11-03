from importlib.metadata import version
from sys import platform

import torch
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# print version of libraries
pkgs = ["transformers",
        "torch",
        "datasets",
        "peft"
        ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

model_name = "openai-community/gpt2"


def collate_and_tokenize(data_batch):
    instruction = data_batch["instruction"][0].replace('"', r'\"')
    inputs = data_batch["input"][0].replace('"', r'\"')
    output = data_batch["output"][0].replace('"', r'\"')

    if inputs == "":
        inputs = instruction
        instruction = ""

    # merging into one prompt for tokenization and training
    prompt = f"""###System: 
    {instruction}
    ###Input:
    {inputs}
    ###Output:
    {output}"""

    encoded = tokenizer(prompt,
                        return_tensors="np",
                        padding="max_length",
                        truncation=True,
                        max_length=400)
    encoded["labels"] = encoded["input_ids"]
    return encoded


# Setting up tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          add_eos_token=True,
                                          padding_side="right",
                                          add_bos_token=True,
                                          use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# load data
train_dataset = load_dataset('json',
                             data_dir='/Users/mehikmat/proj/gen-ai/data/phi-2',
                             split="train[0:1000]")
test_dataset = load_dataset('json',
                            data_dir='/Users/mehikmat/proj/gen-ai/data/phi-2',
                            split="train[1000:1100]")

columns_to_remove = ["instruction", "input", "output"]
tokenized_train_dataset = train_dataset.map(collate_and_tokenize,
                                            batched=True,
                                            batch_size=1,
                                            remove_columns=columns_to_remove)

tokenized_test_dataset = test_dataset.map(collate_and_tokenize,
                                          batched=True,
                                          batch_size=1,
                                          remove_columns=columns_to_remove)

# Configuration to load model in 4-bit quantized
# Loading model with compatible settings
# Note latest bitsandbytes is not supported in mac
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if platform != "darwin":
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device,
    )

model.resize_token_embeddings(len(tokenizer))
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        'c_attn',
        'c_proj',
        'c_fc',
    ],
    inference_mode=False
)
model = get_peft_model(model, peft_config)
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=5,
    gradient_checkpointing=True,
    eval_strategy="steps",
    learning_rate=5e-05,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    num_train_epochs=2,
    group_by_length=True,
    fp16=False,
    push_to_hub=False,
    adam_beta2=0.999,
    do_train=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    peft_config=peft_config,
    args=training_arguments,
    tokenizer=tokenizer,
    max_seq_length=2024
)

torch.cuda.empty_cache()
# start training
trainer.train()

# save trained model
trainer.model.save_pretrained("/Users/mehikmat/proj/gen-ai/model/gpt124M_tuned")
