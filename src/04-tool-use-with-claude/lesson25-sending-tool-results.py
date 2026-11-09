"""Building with the Claude API: Lesson 25 - Sending tool results

We run the tool Claude asked for, send the result back in a tool_result block, and Claude uses it to write its final answer.

The full round trip (lesson 24 did steps 1-2, this lesson adds 3-5):

    1. our code -> Claude:  question + tools
    2. Claude -> our code:  tool_use block   (name, input, id)
    3. our code runs it:    get_current_datetime(**tool_use.input)  ->  "15:04:22"
    4. our code -> Claude:  whole history + tool_result block  (tool_use_id = the tool_use id) + tools
    5. Claude -> our code:  final text answer, using the result

The tool_result block, sent in a USER message:

    {"type": "tool_result", "tool_use_id": <tool_use.id>, "content": "15:04:22", "is_error": False}

Rules:
- tool_use_id must be the id of the tool_use block it answers. With several tool calls, each result
  is matched to its call by id, whatever the order.
- Send the whole conversation again, plus the new tool_result.
- Send the tools again, even if no new tool call is expected: Claude needs the schema to understand
  the tool_use in the history.
- content is a string. If the tool failed, send the error message and is_error=True: Claude sees it
  and can retry with other arguments.

Different from the course: the course picks the block with response.content[1]. On current models a
thinking block usually comes first, so the position changes. Here the tool_use block is found by its
type instead.

https://academy.claude.com/courses/building-with-the-claude-api/sending-tool-results

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from datetime import datetime
import anthropic
from anthropic.types import ToolParam
from helpers.models import CHEAPEST
from helpers.log_helper import (
    log_turn, log_model_request, log_model_response, log_tool_request, log_tool_response, log_answer_to_user,
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


def run_tool(tool_use):
    """Runs the tool function Claude asked for and returns the tool_result block to send back"""
    try:
        result = get_current_datetime(**tool_use.input)
        is_error = False
    except Exception as error:
        # Claude sees the message and can retry with other arguments.
        result = f"Error: {error}"
        is_error = True

    return {
        "type": "tool_result",
        "tool_use_id": tool_use.id,
        "content": result,
        "is_error": is_error,
    }


def main():
    messages = []
    messages.append({"role": "user", "content": user_prompt})

    # Steps 1-2: the question and the tool go to Claude; Claude asks for the tool (lesson 24).
    log_turn(1, note="question + tool schema")
    log_model_request(model=model, max_tokens=max_tokens, messages=messages, tools=[get_current_datetime_schema])
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=[get_current_datetime_schema],
    )
    log_model_response(response)
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        # Claude answered without the tool, so there is nothing to run.
        log_answer_to_user(response)
        return

    # Step 3: run the tool. Find the tool_use block by its type: its position varies.
    tool_use = next(block for block in response.content if block.type == "tool_use")
    log_tool_request(tool_use.name, tool_use.input)
    tool_result = run_tool(tool_use)
    log_tool_response(tool_use.name, tool_result["content"])

    # Step 4: the result goes back in a user message, matched to the call by tool_use_id.
    messages.append({"role": "user", "content": [tool_result]})

    log_turn(2, note="history + tool result + tool schema")
    log_model_request(model=model, max_tokens=max_tokens, messages=messages, tools=[get_current_datetime_schema])
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=[get_current_datetime_schema],
    )
    log_model_response(response)

    # Step 5: Claude's final answer, written with the tool result.
    log_answer_to_user(response)


if __name__ == "__main__":
    main()
