import json

# fine tuning
file_path = "/Users/mehikmat/proj/gen-ai/data/instruction-data.json"

with open(file_path, "r") as file:
    data = json.load(file)
print("Number of entries:", len(data))

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
