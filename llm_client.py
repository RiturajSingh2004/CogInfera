"""
CogInfera — Unified LLM Client (OpenRouter / Qwen 3.5-9B)
"""

import json
import re
from openai import OpenAI
import config


class LLMClient:
    """Wrapper around OpenRouter API for all LLM calls."""

    def __init__(self):
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )
        self.model = config.LLM_MODEL

    # ── Core chat ───────────────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        reasoning: bool = True,
    ) -> str:
        """Send messages and return the assistant's text reply."""
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        extra = {}
        if reasoning and config.LLM_REASONING_ENABLED:
            extra["reasoning"] = {"enabled": True}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            extra_body=extra if extra else None,
        )
        return response.choices[0].message.content

    # ── JSON‑structured output ──────────────────────────────────────
    def chat_json(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        reasoning: bool = True,
    ) -> dict | list:
        """Chat and parse the reply as JSON.

        The system prompt should instruct the model to respond with valid JSON.
        Falls back to extracting JSON from markdown fences if needed.
        """
        raw = self.chat(messages, system_prompt=system_prompt, reasoning=reasoning)
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try extracting from ```json ... ``` fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"LLM did not return valid JSON.\nRaw response:\n{raw}")
