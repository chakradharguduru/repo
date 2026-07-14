"""
router.py — decides what to do with a user message BEFORE retrieval.

This is the piece that was missing. Two jobs, one cheap LLM call:

1. mode:
     "general"  -> greeting, small talk, math (1+1), or a definition of a
                   general concept. Answer directly, DO NOT search the repo.
     "codebase" -> a question about the AXE repo (files, pipelines,
                   transformations, tables, logic, config). Retrieve.

2. standalone_query (only when mode == "codebase"):
     Rewrite a follow-up into a self-contained search query by resolving
     pronouns / implicit references using the conversation history.

       history mentions survey_summary.py ...
       user now asks: "what is the table name"
       ->  "what table does survey_summary.py write data into"

     This is what stops the retriever from latching onto a stray generic
     word (like "table") and pulling the wrong files.
"""
import json
from typing import Dict, List

from groq import Groq

import config


_ROUTER_SYSTEM = (
    "You are a query router for a code assistant that answers questions about "
    "the AXE Data Platform repository. You return ONLY a JSON object."
)

_ROUTER_TEMPLATE = """
Decide how to handle the user's CURRENT message.

Return ONLY a JSON object, no prose, no markdown fences. Schema:
{{
  "mode": "codebase" | "general",
  "standalone_query": "<string>"
}}

Rules:
- "codebase": the user is asking about this repository — its files, code,
  pipelines, transformations, tables, joins, config, or anything they were
  already discussing about the code. Use history to know if a short message
  like "what is the table name" is a follow-up about the code.
- "general": greetings, thanks, small talk, math (e.g. 1+1), or definitions
  of general concepts NOT specific to this repo. These need no code search.
- standalone_query: only when mode is "codebase". Rewrite the CURRENT message
  into ONE self-contained search query by pulling the relevant file/subject
  from the conversation history. If it is already self-contained, repeat it
  unchanged. When mode is "general", set standalone_query to "".

Examples:

History: (none)
Current: hi there
-> {{"mode": "general", "standalone_query": ""}}

History: (none)
Current: what is 2 + 2
-> {{"mode": "general", "standalone_query": ""}}

History:
USER: what transformations are done in survey summary
ASSISTANT: survey_summary.py applies ... and writes the result ...
Current: what is the table name
-> {{"mode": "codebase", "standalone_query": "what table name does survey_summary.py write its data into"}}

History:
USER: where does the givex transform join brand
ASSISTANT: In givex.py the brand is joined ...
Current: what is RAG
-> {{"mode": "general", "standalone_query": ""}}

Conversation History:
{history}

Current message:
{question}
""".strip()


def _render_history(history) -> str:
    if not history:
        return "(none)"
    lines: List[str] = []
    for m in history[-8:]:
        # history items may be pydantic ChatMessage objects or plain dicts
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role is None and isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        lines.append(f"{str(role).upper()}: {content}")
    return "\n".join(lines)


def route_query(query: str, history=None) -> Dict:
    """Return {"mode": "...", "standalone_query": "..."}. Never raises."""
    if not config.GROQ_API_KEY:
        # No key -> safest default is to search with the raw query.
        return {"mode": "codebase", "standalone_query": query}

    client = Groq(api_key=config.GROQ_API_KEY)
    prompt = _ROUTER_TEMPLATE.format(
        history=_render_history(history),
        question=query,
    )

    try:
        resp = client.chat.completions.create(
            model=config.ROUTER_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
    except Exception as e:
        if config.DEBUG:
            print("Router failed, defaulting to codebase:", e)
        return {"mode": "codebase", "standalone_query": query}

    mode = data.get("mode", "codebase")
    if mode not in ("codebase", "general"):
        mode = "codebase"

    standalone = (data.get("standalone_query") or "").strip()
    if mode == "codebase" and not standalone:
        standalone = query  # fall back to the raw question

    if config.DEBUG:
        print(f"\nRouter -> mode={mode}")
        if mode == "codebase":
            print(f"Router -> standalone_query: {standalone}")

    return {"mode": mode, "standalone_query": standalone}