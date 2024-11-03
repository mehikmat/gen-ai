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
    Answer the following question based on the given input.
    ###Input:
    "freind --> friend",
    ###Question:
    Evaluate the following phrase by transforming it into the spelling given.
    ###Answer:
    The spelling of the given phrase "freind" is incorrect, the correct spelling is "friend"."""
encoded = tokenizer(prompt,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=False,
                    truncation=True)

response = model.generate(**encoded, max_length=1000)
result = tokenizer.batch_decode(response, skip_special_tokens=True)
print(result)
