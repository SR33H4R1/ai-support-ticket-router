from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request
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


app = Flask(__name__)
ticket_chain = build_chain()


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.post("/ticket")
def ticket() -> Any:
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    user_name = payload.get("user_name")
    message = payload.get("message")

    if not isinstance(user_name, str) or not user_name.strip():
        return jsonify({"error": "Field 'user_name' is required."}), 400

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Field 'message' is required."}), 400

    result = ticket_chain.invoke(payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

