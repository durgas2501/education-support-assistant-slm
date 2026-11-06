
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="float32"
)

print("Model loaded!")

system_prompt = """
You are an Education Support Assistant.
Answer students' educational questions.
Use simple English.
Give short, clear and easy-to-understand answers.
If possible, include a simple example.
"""

with open("data/education_dataset.jsonl", "r") as file:
    dataset = [json.loads(line) for line in file if line.strip()]

results = []

for item in dataset:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["question"]}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    start_time = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    latency = round(time.time() - start_time, 2)

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    results.append({
        "question": item["question"],
        "answer": answer,
        "latency_seconds": latency
    })

    print("\nQuestion:", item["question"])
    print("Adapted Answer:", answer)
    print("Time:", latency, "seconds")

with open("reports/adapted_results.jsonl", "w") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")

print("\nTotal Questions:", len(results))
print("Adapted results saved successfully!")
