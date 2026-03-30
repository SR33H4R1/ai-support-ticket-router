# AI Support Ticket Router

AI Support Ticket Router is a small Flask + LangChain project that classifies support tickets and routes them to either a billing or technical support response flow using a local Ollama model.

## What it does

1. Accepts a support ticket as JSON.
2. Uses structured output to classify the ticket as `billing` or `tech`.
3. Generates a department-specific response.
4. Returns both the classification and the final response as JSON.

## Tech stack

- Python
- Flask
- LangChain
- Ollama

## Project structure

```text
.
|-- app.py
|-- client_example.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the default model is available:

```bash
ollama pull llama3.1
```

## Run the API

```bash
python app.py
```

## Run the sample client

```bash
python client_example.py
```

## API

### `GET /health`

Returns a simple status response.

### `POST /ticket`

Request body:

```json
{
  "user_name": "Sreehari",
  "message": "My app keeps crashing after the latest update.",
  "history": [
    ["human", "Hi"],
    ["ai", "Hello, how can I help you today?"]
  ]
}
```

Example response:

```json
{
  "user_name": "Sreehari",
  "message": "My app keeps crashing after the latest update.",
  "classification": {
    "department": "tech",
    "reason": "The ticket describes an application stability issue after an update."
  },
  "response": "Technical Assist AI: Hi Sreehari, here are the first steps to isolate the crash..."
}
```

## Notes

- This is a learning project, not a production support system.
- The classifier only routes between two departments: billing and tech.
- Output quality depends on the local model.

## Possible improvements

- add more departments
- persist ticket history
- add automated API tests
- expose the API with a simple frontend
