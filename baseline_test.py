
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

print("Model loaded successfully!")

# Read dataset
with open("data/education_dataset.jsonl", "r") as file:
    dataset = [json.loads(line) for line in file if line.strip()]

results = []

# Test each question
for item in dataset:
    question = item["question"]

    messages = [
        {"role": "user", "content": question}
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

    end_time = time.time()

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    latency = round(end_time - start_time, 2)

    results.append({
        "question": question,
        "expected_answer": item.get("answer", ""),
        "generated_answer": answer,
        "latency_seconds": latency
    })

    print("\nQuestion:", question)
    print("Answer:", answer)
    print("Time:", latency, "seconds")

# Save baseline results
with open("reports/baseline_results.jsonl", "w") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")

print("\nTotal Questions:", len(results))
print("Baseline results saved successfully!")
