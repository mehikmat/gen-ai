import json


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# fine tuning
file_path = "/data/phi-2/instruction-data.json"

with open(file_path, "r") as file:
    data = json.load(file)
print("Number of entries:", len(data))

# format data in alpaca format
for d in data:
    d["text"] = format_input(d)

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.15)  # 15% for testing

train_data = data[:train_portion]
test_data = data[train_portion:]

print("Training set length:", len(train_data))
print("Test set length:", len(test_data))

with open("train.json", "w") as json_file:
    json.dump(train_data, json_file, indent=4)

with open("test.json", "w") as json_file:
    json.dump(test_data, json_file, indent=4)
