import json

from litgpt import LLM

with open("test.json", "r") as file:
    test_data = json.load(file)

print("Input: " + str(test_data[0]))

llm1 = LLM.load("microsoft/phi-2")
response1 = llm1.generate(test_data[0])
print("Before fine tune: " + response1)

llm2 = LLM.load("out/finetune/lora/final/")
response2 = llm2.generate(test_data[0])
print("After fine tune: " + response2)
