
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Education Support Assistant")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading SLM model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="float32"
)

print("SLM model loaded successfully!")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Education Support Assistant is running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    messages = [
        {
            "role": "system",
            "content": "You are an Education Support Assistant. Give short, clear and simple English answers."
        },
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return {
        "question": question,
        "answer": answer
    }
