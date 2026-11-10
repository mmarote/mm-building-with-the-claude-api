"""Building with the Claude API: Lesson 24 - Handling message blocks

We send a question together with the tool schema, and Claude answers with a message made of
several blocks, one of them a request to run our tool.

What happens in this lesson:

    our code:  "What is the exact time, formatted as HH:MM:SS?"  + tools=[get_current_datetime_schema]
    Claude:    one assistant message, several content blocks:
                 text      "I'll check the current time for you."   (for people to read)
                 tool_use  name="get_current_datetime", input={"date_format": "%H:%M:%S"}, id="toolu_..."

The fields of a tool_use block:
- id:    identifies this tool call. Lesson 25 sends the result back with the same id.
- name:  which tool function to run.
- input: the arguments, as a dictionary.
- type:  "tool_use".

Current models may also return a thinking block before the others (empty by default). It is part of
the message too.

Keeping the conversation:
- Claude does not remember anything between requests, so our code keeps the messages list.
- Append the assistant message with ALL its blocks (content=response.content), not only its text.
  The tool_use block must stay in the history for the next request (lesson 25).

Not done yet: running the tool and sending the result back. That is lesson 25.

The request is sent with client.messages.create directly: chat() from chat_helper returns only the
text, and here we need every block.

https://academy.claude.com/courses/building-with-the-claude-api/handling-message-blocks

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from datetime import datetime
import anthropic
from anthropic.types import ToolParam
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


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


def main():
    messages = []
    messages.append({"role": "user", "content": user_prompt})

    log_turn(1, note="question + tool schema")
    log_model_request(model=model, max_tokens=max_tokens, messages=messages, tools=[get_current_datetime_schema])

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=[get_current_datetime_schema],
    )

    log_model_response(response)

    # Keep the whole assistant message, every block, in the conversation.
    messages.append({"role": "assistant", "content": response.content})

    print(f"\nClaude's message has {len(response.content)} blocks (stop_reason: {response.stop_reason}):")
    for number, block in enumerate(response.content, start=1):
        if block.type == "text":
            print(f"  {number}. text:      {block.text}")
        elif block.type == "tool_use":
            print(f"  {number}. tool_use:  name={block.name}  input={block.input}  id={block.id}")
        else:
            print(f"  {number}. {block.type}")

    print(f"\nThe conversation now has {len(messages)} messages: {[message['role'] for message in messages]}")


if __name__ == "__main__":
    main()
