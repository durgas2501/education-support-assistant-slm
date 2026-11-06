import json

file_path = "data/education_dataset.jsonl"

with open(file_path, "r") as file:
    data = [json.loads(line) for line in file if line.strip()]

print("Dataset loaded successfully!")
print("Total Questions:", len(data))

for item in data:
    print("Question:", item["question"])
    print("Answer:", item["answer"])
    print("Total Questions:", len(data))