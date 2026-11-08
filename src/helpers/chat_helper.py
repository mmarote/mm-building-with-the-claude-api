"""
Shared helpers for sending a conversation to Claude.

The early lessons write these functions out in full, because the lesson is about them.
Later lessons import them from here, so the lesson can focus on its own topic.

    add_user_message(messages, text)        append a user turn to the list
    add_assistant_message(messages, text)   append an assistant turn to the list
    send(messages, ...)                     call Claude, return the whole response object
    chat(messages, ...)                     call Claude, return only the text of the answer

Usage:
    from helpers.chat_helper import add_user_message, add_assistant_message, chat

    messages = []
    add_user_message(messages, "Define quantum computing in one sentence")
    answer = chat(messages)
    add_assistant_message(messages, answer)

Not included: temperature. Models released after Claude Opus 4.6 reject it (see lesson 6).
Assistant prefill is still possible through add_assistant_message, but it is deprecated on current
models (see lessons 8 and 11).

The log_* calls come from helpers/log_helper.py. Turn them off with `log_helper.ENABLED = False`.
"""

import anthropic
from helpers.models import CHEAPEST
from helpers.log_helper import log_model_request, log_model_response


client = anthropic.Anthropic()


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def send(messages, model=CHEAPEST, max_tokens=1000, system=None, stop_sequences=None):
    """Send the conversation to Claude and return the whole response object."""
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }

    # The API does not accept system=None or stop_sequences=None, so they are only added when given.
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    log_model_request(model=model, max_tokens=max_tokens, system=system, messages=messages)

    response = client.messages.create(**params)

    log_model_response(response)

    return response


def chat(messages, model=CHEAPEST, max_tokens=1000, system=None, stop_sequences=None):
    """Send the conversation to Claude and return the text of the answer."""

    response = send(messages, model=model, max_tokens=max_tokens, system=system, stop_sequences=stop_sequences)

    # The answer can come in several text blocks; current models may also return thinking blocks
    # (empty by default), which are skipped here.
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        # Current models think before they answer, and thinking counts toward max_tokens. If thinking
        # uses up max_tokens, the reply has no text at all (stop_reason "max_tokens"). A refusal
        # (stop_reason "refusal") can also come back without text.
        raise RuntimeError(
            f"Claude returned no text (stop_reason: {response.stop_reason}). "
            f"If it is max_tokens, raise max_tokens (now {max_tokens})."
        )
    return text
