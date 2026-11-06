# Education Support Assistant using SLM

## Objective
To build an AI assistant that answers students' educational questions.

## Target Users
School and college students.

## Input
Students' educational questions.

## Output
Simple and understandable answers.

## Success Criteria
The assistant should provide relevant, clear and correct answers.

## Run the FastAPI service

Open a terminal in the project folder and activate the existing virtual environment:

```bash
cd "/Users/machd/Desktop/SLM PROJECT"
source venv/bin/activate
```

If this is a new setup, install the project packages once:

```bash
python -m pip install -r requirements.txt
```

Start the API from the project folder:

```bash
python -m uvicorn src.app:app --reload
```

The first start may take a while because Hugging Face downloads `Qwen/Qwen2.5-0.5B-Instruct` if it is not already cached. When the server says it is ready, open <http://127.0.0.1:8000/docs> to try the API in a browser.

The service provides:

- `GET /health` to check whether the model loaded.
- `POST /ask` to send a question and receive an answer with its generation time.

Example request (in a second terminal):

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Python? Explain in simple words."}'
```

The question must contain 1–2000 characters after trimming spaces. The API rejects empty or oversized questions and unexpected fields. Stop the server with `Ctrl+C`.
## How to Run the Project

1. Activate the virtual environment.
2. Install the required packages.
3. Run the dataset validation:
   python src/check_dataset.py
4. Run the baseline test:
   python src/baseline_test.py
5. Run the adapted test:
   python src/adapted_test.py
6. Run the optimized test:
   python src/optimized_test.py
7. Start the FastAPI application:
   uvicorn src.app:app --reload

Open http://127.0.0.1:8000/docs to test the API.

## Performance Results

| Test | Average Response Time |
|---|---:|
| Baseline | 4.50 seconds |
| Adapted | 4.21 seconds |
| Optimized | 2.47 seconds |

## Technology Used

- Python
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct
- FastAPI
- PyTorch
