"""Building with the Claude API: Lesson 27 - Implementing multiple turns

We complete the conversation loop of lesson 26: run_tools handles every tool request in a
reply, and run_tool picks the function by the tool's name.

The loop (from lesson 26, now with a full run_tools):

    run_conversation
      └─ while True:
           response = send(messages, tools)         Claude replies: text and/or tool requests
           add_assistant_message(messages, response)
           if response.stop_reason != "tool_use":   no tool requested: the answer is final
               break
           run_tools(response)                       one tool_result block per tool_use block
             └─ run_tool(name, input)                picks the function by the tool's name
           add_user_message(messages, tool_results)  then ask Claude again

One reply can ask for several tools at once:

    Claude:   text + tool_use (id A) + tool_use (id B)
    our code: tool_result (tool_use_id A) + tool_result (tool_use_id B), all in ONE user message

Each tool_result block:
- tool_use_id: the id of the tool_use block it answers.
- content: the tool's output, as text (json.dumps).
- is_error: True if the tool failed. The error message goes in content, so Claude can retry.
  Every tool_use block gets a tool_result, even when the tool fails.

To add a tool: write its function and schema, add it to run_tool and to the tools list. The loop
itself does not change (lesson 28 adds a second tool).

The helpers come from helpers/chat_helper.py (lesson 26 writes them out in full). The
course's chat() is send() there: it returns the whole message. chat() in chat_helper returns
only the text.

https://academy.claude.com/courses/building-with-the-claude-api/implementing-multiple-turns

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import json
from datetime import datetime
from anthropic.types import ToolParam
from helpers.chat_helper import add_user_message, add_assistant_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_tool_request, log_tool_response


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


model = CHEAPEST
# Claude thinks before it answers, and thinking counts toward max_tokens; too low a limit can leave
# no answer at all. You only pay for the tokens Claude actually uses.
max_tokens = 16000
# Two pieces of information, so Claude may ask for the tool twice in one reply.
user_prompt = "What is the current time, formatted as HH:MM:SS, and today's date, formatted as YYYY-MM-DD?"


def run_tool(tool_name, tool_input):
    """Runs the tool function with this name; add a branch here for every new tool"""
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    raise ValueError(f"Unknown tool: {tool_name}")


def run_tools(message):
    """Runs every tool Claude asked for in this message; returns one tool_result block per request"""
    tool_requests = [block for block in message.content if block.type == "tool_use"]
    tool_result_blocks = []

    for tool_request in tool_requests:
        log_tool_request(tool_request.name, tool_request.input)
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False,
            }
        except Exception as e:
            # Claude still gets an answer for this request, and can retry with other arguments.
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }
        log_tool_response(tool_request.name, tool_result_block["content"])
        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks


def run_conversation(messages):
    """Asks Claude, runs the tools it asks for, and repeats until Claude stops asking for tools"""
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = send(messages, model=model, max_tokens=max_tokens, tools=[get_current_datetime_schema])
        add_assistant_message(messages, response)

        if response.stop_reason != "tool_use":
            break

        # Claude's text between tool calls (e.g. "First I'll work out the date"). The course prints it
        # before the check above; printing it here shows the final answer only once, in main().
        print(text_from_message(response))

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

    return messages


def main():
    messages = []
    add_user_message(messages, user_prompt)

    messages = run_conversation(messages)

    # The last message is Claude's final answer.
    print(f"\n=== ANSWER TO USER ===\n{text_from_message(messages[-1])}")


if __name__ == "__main__":
    main()
