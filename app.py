from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.1"

CLASSIFIER_SCHEMA = {
    "title": "support_ticket_classifier",
    "type": "object",
    "properties": {
        "department": {
            "type": "string",
            "description": "Route the ticket to either 'billing' or 'tech'.",
        },
        "reason": {
            "type": "string",
            "description": "One short sentence explaining why the ticket belongs to that department.",
        },
    },
    "required": ["department", "reason"],
}



class HistoryItem(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in {"human", "ai"}:
            raise ValueError("role must be 'human' or 'ai'")
        return v


class TicketRequest(BaseModel):
    user_name: str
    message: str
    history: list[HistoryItem] = []

    @field_validator("user_name", "message")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank")
        return v.strip()


class ClassificationResult(BaseModel):
    department: str
    reason: str


class TicketResponse(BaseModel):
    user_name: str
    message: str
    classification: ClassificationResult
    response: str


class HealthResponse(BaseModel):
    status: str
    model: str



def normalize_history(payload: dict[str, Any]) -> list[tuple[str, str]]:
    raw_history = payload.get("history", [])
    normalized: list[tuple[str, str]] = []

    for item in raw_history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            role, content = item
        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        else:
            continue

        if role in {"human", "ai"} and isinstance(content, str) and content.strip():
            normalized.append((role, content.strip()))

    return normalized


def build_chain(model_name: str = MODEL_NAME):
    llm = ChatOllama(model=model_name)
    parser = StrOutputParser()
    classifier_llm = llm.with_structured_output(CLASSIFIER_SCHEMA)

    classifier_prompt = PromptTemplate.from_template(
        "You route customer support tickets.\n"
        "Choose exactly one department: billing or tech.\n"
        "Billing handles payments, refunds, subscriptions, invoices, and charges.\n"
        "Tech handles bugs, crashes, login issues, app behavior, performance, and troubleshooting.\n"
        "Return structured output using the provided schema.\n\n"
        "Ticket:\n{message}"
    )

    billing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a billing support specialist. Respond professionally and clearly.\n"
                "Start with 'Billing Assist AI:'.\n"
                "Address the user by name.\n"
                "Give a direct explanation and one or two practical next steps.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{message}"),
        ]
    )

    tech_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a technical support specialist. Respond professionally and clearly.\n"
                "Start with 'Technical Assist AI:'.\n"
                "Address the user by name.\n"
                "Give numbered troubleshooting steps when appropriate.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{message}"),
        ]
    )

    classifier_chain = classifier_prompt | classifier_llm
    billing_chain = billing_prompt | llm | parser
    tech_chain = tech_prompt | llm | parser

    def route_ticket(state: dict[str, Any]) -> dict[str, Any]:
        department = state["classification"]["department"].strip().lower()
        response_chain = billing_chain if department == "billing" else tech_chain
        response = response_chain.invoke(state)

        return {
            "user_name": state["user_name"],
            "message": state["message"],
            "classification": state["classification"],
            "response": response,
        }

    prepared_input = RunnablePassthrough.assign(
        user_name=RunnableLambda(lambda x: x["user_name"].strip()),
        message=RunnableLambda(lambda x: x["message"].strip()),
        history=RunnableLambda(normalize_history),
    )

    return (
        prepared_input
        | RunnablePassthrough.assign(classification=classifier_chain)
        | RunnableLambda(route_ticket)
    )



app = FastAPI(
    title="AI Support Ticket Router",
    description="Classifies support tickets and routes them to billing or technical departments using LLMs.",
    version="1.0.0",
)

ticket_chain = build_chain()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model=MODEL_NAME)


@app.post("/ticket", response_model=TicketResponse)
async def create_ticket(ticket: TicketRequest):
    try:
        payload = {
            "user_name": ticket.user_name,
            "message": ticket.message,
            "history": [{"role": h.role, "content": h.content} for h in ticket.history],
        }
        result = ticket_chain.invoke(payload)
        return TicketResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
