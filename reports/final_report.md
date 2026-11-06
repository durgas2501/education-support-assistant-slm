# Education Support Assistant Using Small Language Model (SLM)

## 1. Introduction
The Education Support Assistant is a project developed using a Small Language Model (SLM). It helps students understand educational topics by providing short, clear, and simple answers.

## 2. Use Case
- **Target Users:** School and college students.
- **Input:** Educational questions.
- **Output:** Simple English answers.
- **Goal:** To provide relevant and easy-to-understand explanations.

## 3. Model Research and Selection
Three Small Language Models were considered:

| Model | Parameters | Context Length |
|---|---:|---:|
| SmolLM2-360M-Instruct | 360M | 8K |
| Qwen2.5-0.5B-Instruct | 0.5B | 32K |
| TinyLlama-1.1B-Chat | 1.1B | 2K |

**Selected Model:** Qwen2.5-0.5B-Instruct.

It was selected because it is a compact instruction-tuned model with a 32K context length and an Apache 2.0 license.

## 4. Dataset Preparation
A small educational question-and-answer dataset was created in JSONL format.

- Total records: 10
- Topics include Python, AI, Machine Learning, variables, functions, databases, HTML, computer networks, neural networks, and cloud computing.
- The dataset was checked using Python.

## 5. Baseline Testing
The selected model was loaded using Hugging Face Transformers. It was tested with 10 educational questions using the original prompt.

**Average response time: 4.50 seconds.**

## 6. Model Adaptation
A system prompt was added to guide the model to behave as an Education Support Assistant.

The prompt instructed the model to provide simple English and short, clear answers.

This was prompt-based adaptation. Model weight fine-tuning was not performed.

**Average response time: 4.21 seconds.**

## 7. Model Optimization
The generation settings were optimized by:
- Reducing maximum new tokens from 100 to 60.
- Using short-answer instructions.
- Setting deterministic generation with `do_sample=False`.

**Average response time: 2.47 seconds.**

The observed response time was approximately 45% lower than the baseline. Shorter responses may sometimes provide less detail.

## 8. Deployment
The application was deployed using FastAPI.

The API contains:
- **GET /** – Checks whether the application is running.
- **POST /ask** – Accepts a question and returns an answer.

Input validation was added to reject empty questions. The API was tested successfully using Swagger UI.

## 9. Limitations
- The dataset contains only 10 records.
- Model weight fine-tuning was not performed.
- Response quality was not independently evaluated using a human assessment.
- Response times may vary depending on hardware and system conditions.
- The model may sometimes generate incorrect or incomplete answers.

## 10. Conclusion
The project demonstrates how a Small Language Model can be used to build an Education Support Assistant. The Qwen2.5-0.5B-Instruct model was tested, adapted through prompting, optimized, and deployed using FastAPI.

The project helped demonstrate the basic workflow of SLM inference, prompt-based adaptation, performance testing, optimization, and API deployment.

## 11. Future Improvements
- Increase the size and quality of the educational dataset.
- Perform LoRA or QLoRA fine-tuning when suitable data and hardware are available.
- Evaluate answer quality using a structured test set.
- Improve response validation and application monitoring.