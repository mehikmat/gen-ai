from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

instruction_text = f"""Below is an instruction that describes a task.
Write a response that appropriately completes the request."""

model_name = "openai-community/gpt2"
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name)

model = PeftModel.from_pretrained(base_model, "/Users/mehikmat/proj/gen-ai/model/gpt124M_tuned")
model = model.merge_and_unload()

# Setting up tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          add_eos_token=True,
                                          padding_side="right",
                                          add_bos_token=True,
                                          use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
prompt = f"""###System:
    What are the first 10 square numbers?
    ###Input:
    ###Output:
    """
encoded = tokenizer(prompt,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=False,
                    truncation=True)

response = model.generate(**encoded, max_length=1000)
result = tokenizer.batch_decode(response, skip_special_tokens=True)
print(result)
print(tokenizer.batch_decode(base_model.generate(**encoded, max_length=1000), skip_special_tokens=True))
