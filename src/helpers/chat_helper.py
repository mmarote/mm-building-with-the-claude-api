"""
Shared helpers for sending a conversation to Claude.

The early lessons write these functions out in full, because the lesson is about them.
Later lessons import them from here, so the lesson can focus on its own topic.

    add_user_message(messages, message)       append a user turn: text, a list of blocks, or a Message
    add_assistant_message(messages, message)  append an assistant turn: text, a list of blocks, or a Message
    send(messages, ..., tools=None)           call Claude, return the whole Message (all its blocks)
    chat(messages, ...)                       call Claude, return only the text of the answer
    text_from_message(message)                the text of a Message (or of a message in the list)

Usage:
    from helpers.chat_helper import add_user_message, add_assistant_message, chat

    messages = []
    add_user_message(messages, "Define quantum computing in one sentence")
    answer = chat(messages)
    add_assistant_message(messages, answer)

With tools (lesson 26 writes these out in full), keep whole messages, not only their text:

    response = send(messages, tools=[...])
    add_assistant_message(messages, response)          every block, including tool_use
    add_user_message(messages, tool_result_blocks)     a list of tool_result blocks
    print(text_from_message(response))

Not included: temperature. Models released after Claude Opus 4.6 reject it (see lesson 6).
Assistant prefill is still possible through add_assistant_message, but it is deprecated on current
models (see lessons 8 and 11).

The log_* calls come from helpers/log_helper.py. Turn them off with `log_helper.ENABLED = False`.
"""

import anthropic
from anthropic.types import Message
from helpers.models import CHEAPEST
from helpers.log_helper import log_model_request, log_model_response


client = anthropic.Anthropic()


def add_user_message(messages, message):
    """Append a user turn: plain text, a list of blocks (e.g. tool_result), or a Message's blocks."""
    messages.append({"role": "user", "content": message.content if isinstance(message, Message) else message})


def add_assistant_message(messages, message):
    """Append an assistant turn: plain text, a list of blocks, or a Message (keeping every block)."""
    messages.append({"role": "assistant", "content": message.content if isinstance(message, Message) else message})


def send(messages, model=CHEAPEST, max_tokens=1000, system=None, stop_sequences=None, tools=None):
    """Send the conversation to Claude and return the whole Message, with all its blocks."""
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }

    # The API does not accept None for these, so they are only added when given.
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    if tools:
        params["tools"] = tools

    log_model_request(model=model, max_tokens=max_tokens, system=system, messages=messages, tools=tools)

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


def text_from_message(message):
    """The text of a message (a Message, or one from the messages list): its text blocks joined."""
    content = message.content if isinstance(message, Message) else message["content"]
    return "\n".join(block.text for block in content if block.type == "text")
