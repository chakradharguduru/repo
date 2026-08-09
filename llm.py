"""
llm.py — the generation step.

Two entry points now:
- generate_answer(): grounded answer for CODEBASE questions (uses retrieved
  chunks + full-file context, exactly as before).
- answer_general(): plain answer for GENERAL questions (greetings, math,
  concept definitions). No repo context, no sources.
"""

from typing import List, Dict
from groq import Groq

import config
from context import build_context

SYSTEM_PROMPT = (
    "You are an expert assistant for the AXE Data Platform, grounded in both "
    "its repository code and its Confluence documentation."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer the user's question directly. "
    "This question is NOT about the AXE code repository, so do not mention files "
    "or code unless the user brings them up."
)

USER_PROMPT_TEMPLATE = """
You are a knowledge assistant that answers questions using a codebase and its
Confluence documentation.

- Answer ONLY using the material shown below (repository code and/or
  Confluence pages).
- Use the conversation history to understand follow-up questions such as
  "Where is it joined?" or "What happens next?".
- Never invent code or documentation that is not present below.
- Always cite which file(s) or documentation page(s) (by title/URL) your
  answer is based on.
- If the material below does not contain enough information, say so.

Conversation History:
{history}

Current Question:
{question}

Retrieved Material:
{context}
"""


def _get_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return Groq(api_key=config.GROQ_API_KEY)


def _history_to_messages(history) -> List[Dict]:
    """Turn stored history (ChatMessage objects or dicts) into API messages."""
    messages = []
    for m in (history or [])[-8:]:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role is None and isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


# ---------------------------------------------------------------------
# GENERAL questions — no retrieval
# ---------------------------------------------------------------------
def answer_general(query: str, history=None) -> str:
    client = _get_client()

    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    messages.extend(_history_to_messages(history))
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=config.GROQ_TEMPERATURE,
        messages=messages,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# CODEBASE questions — grounded in retrieved chunks (unchanged behavior)
# ---------------------------------------------------------------------
def generate_answer(query: str, chunks: List[Dict], history=None) -> str:
    if not chunks:
        return (
            "I couldn't find any relevant code for that question "
            "in the indexed repository."
        )

    client = _get_client()
    context_block = build_context(chunks)

    if not context_block:
        return "I found matching chunks, but couldn't read their source files."

    conversation = ""
    if history:
        conversation = "Previous Conversation:\n"
        for message in history:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None and isinstance(message, dict):
                role = message.get("role", "")
                content = message.get("content", "")
            conversation += f"{str(role).upper()}: {content}\n"
        conversation += "\n"

    user_msg = USER_PROMPT_TEMPLATE.format(
        history=conversation,
        question=query,
        context=context_block,
    )

    print("\nPrompt Length:", len(user_msg))
    print("=" * 80)
    print(user_msg[:4000])
    print("=" * 80)
    print("Model       :", config.GROQ_MODEL)
    print("Temperature :", config.GROQ_TEMPERATURE)

    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(user_msg)
    print("Prompt saved to prompt.txt")

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=config.GROQ_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    print("\nLLM Response")
    print("=" * 80)
    print(response.choices[0].message.content)

    return response.choices[0].message.content