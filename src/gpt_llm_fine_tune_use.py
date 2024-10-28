import json

from litgpt import LLM

with open("test.json", "r") as file:
    test_data = json.load(file)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


print("Input: " + str(test_data[0]))

llm = LLM.load("microsoft/phi-2")
response1 = llm.generate(format_input(test_data[0]))
print("Before fine tune: " + response1)

del llm
llm2 = LLM.load("out/finetune/lora/final/")
response2 = llm2.generate(format_input(test_data[0]))
print("After fine tune: " + response2)
