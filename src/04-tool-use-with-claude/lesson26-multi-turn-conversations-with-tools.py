"""Building with the Claude API: Lesson 26 - Multi-turn conversations with tools

Claude may need several tools, one after another, to answer one question, so our code runs a
conversation loop until Claude stops asking for tools.

Example: "What day is 103 days from today?"

    1. user asks                       4. Claude asks for add_duration_to_datetime
    2. Claude asks for get_current_datetime    5. our code runs it, sends the result
    3. our code runs it, sends the result      6. Claude has all it needs: final answer

The conversation loop (the heart of this lesson):

    def run_conversation(messages):
        while True:
            response = chat(messages, tools=[...])
            add_assistant_message(messages, response)

            if response.stop_reason != "tool_use":      # Claude is not asking for a tool:
                break                                   # its answer is final

            tool_result_blocks = run_tools(response)    # run every tool it asked for
            add_user_message(messages, tool_result_blocks)

        return messages

The helpers, updated so the loop can pass whole messages around:

    chat(messages, tools=...)            returns the whole Message (all blocks), not only its text
    add_assistant_message(messages, m)   accepts a Message: keeps every block in the history
    add_user_message(messages, m)        accepts text, or a list of blocks (e.g. tool_result blocks)
    text_from_message(m)                 joins the text blocks, for the final answer

Here there is only one tool, get_current_datetime, so run_tools is minimal. Lesson 27 completes
run_tools and adds run_tool, which picks the function by the tool's name.

Different from the course:
- The course writes the loop as pseudo-code; here it runs. "if response isn't asking for a tool"
  becomes a check of stop_reason, and chat() gets the tools (the course's sketch leaves them out).
- chat() here works like send() in helpers/chat_helper.py; the shared chat() still returns
  only text.

https://academy.claude.com/courses/building-with-the-claude-api/multi-turn-conversations-with-tools

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from datetime import datetime
import anthropic
from anthropic.types import Message, ToolParam
from helpers.models import CHEAPEST
from helpers.log_helper import (
    log_turn, log_model_request, log_model_response, log_tool_request, log_tool_response,
)


# The tool function (lesson 22) and its schema (lesson 23).
def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time, formatted with date_format (strftime codes)"""

    if not date_format:
        raise ValueError("date_format cannot be empty")
    
    return datetime.now().strftime(date_format)


# The tool schema: how Claude learns about get_current_datetime.
get_current_datetime_schema = ToolParam({
    "name": "get_current_datetime",
    "description": "Returns the current date and time formatted according to the specified format",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "A string specifying the format of the returned datetime. Uses Python's strftime format codes.",
                "default": "%Y-%m-%d %H:%M:%S",
            }
        },
        "required": [],
    },
})



client = anthropic.Anthropic()

model = CHEAPEST
# Claude thinks before it answers, and thinking counts toward max_tokens; too low a limit can leave
# no answer at all. You only pay for the tokens Claude actually uses.
max_tokens = 16000
user_prompt = "What is the exact time, formatted as HH:MM:SS?"


# ── The updated helpers ──────────────────────────────────────────────────────

def add_user_message(messages, message):
    """Appends a user turn: plain text, a list of blocks (e.g. tool_result), or a Message's blocks"""
    messages.append({
        "role": "user",
        "content": message.content if isinstance(message, Message) else message,
    })


def add_assistant_message(messages, message):
    """Appends an assistant turn, keeping every block when given a Message"""
    messages.append({
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message,
    })


def chat(messages, system=None, stop_sequences=None, tools=None):
    """Sends the conversation to Claude and returns the whole Message, with all its blocks"""
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    # The API does not accept None for these, so they are only added when given.
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    if tools:
        params["tools"] = tools

    log_model_request(model=model, max_tokens=max_tokens, messages=messages, system=system, tools=tools)
    message = client.messages.create(**params)
    log_model_response(message)
    return message


def text_from_message(message):
    """The text of a message (a Message, or one from the messages list): its text blocks joined"""
    content = message.content if isinstance(message, Message) else message["content"]
    return "\n".join(block.text for block in content if block.type == "text")


# ── The conversation loop ────────────────────────────────────────────────────

def run_tools(message):
    """Runs every tool Claude asked for in this message; returns the tool_result blocks to send back"""
    tool_result_blocks = []
    for tool_use in [block for block in message.content if block.type == "tool_use"]:
        log_tool_request(tool_use.name, tool_use.input)
        try:
            # Only one tool exists so far. Lesson 27 picks the function by tool_use.name.
            result = get_current_datetime(**tool_use.input)
            is_error = False
        except Exception as error:
            # Claude sees the message and can retry with other arguments.
            result = f"Error: {error}"
            is_error = True
        log_tool_response(tool_use.name, result)

        tool_result_blocks.append({
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": result,
            "is_error": is_error,
        })
    return tool_result_blocks


def run_conversation(messages):
    """Asks Claude, runs the tools it asks for, and repeats until Claude stops asking for tools"""
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = chat(messages, tools=[get_current_datetime_schema])

        add_assistant_message(messages, response)

        # Claude is not asking for a tool: its answer is final.
        if response.stop_reason != "tool_use":
            break

        tool_result_blocks = run_tools(response)
        add_user_message(messages, tool_result_blocks)

    return messages


def main():
    messages = []
    add_user_message(messages, user_prompt)

    messages = run_conversation(messages)

    # The last message is Claude's final answer.
    print(f"\n=== ANSWER TO USER ===\n{text_from_message(messages[-1])}")
    print(f"\nThe conversation has {len(messages)} messages: {[message['role'] for message in messages]}")


if __name__ == "__main__":
    main()
