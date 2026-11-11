"""Building with the Claude API: Lesson 65 - Agents and tools

An agent gets a goal and a set of tools, and Claude decides which tools to use and in what
order; here the same three tools answer four different requests, each in its own way.

Workflow vs. agent:

    workflow   our code fixes the steps      step 1 -> step 2 -> step 3      (lessons 61-64)
    agent      Claude picks the steps        goal + tools -> ? -> ? -> done  (this lesson)

The tools (lesson 28) and the four requests (from the course):

    get_current_datetime   add_duration_to_datetime   set_reminder

    "What time is it?"                               one tool
    "What day of the week will it be in 11 days?"    two tools, chained
    "Remind me to go to the gym next Wednesday"      all three
    "When does my 1-year warranty expire?"           none: Claude asks for the date first

The code is the same tool loop as lessons 27-28: nothing in it says which tool comes when.
The plan comes from Claude, from the goal and the tool descriptions. That makes it an agent,
and also less predictable than a workflow (lesson 67).

Good agent tools are small and combinable (the course's point about Claude Code: bash, read,
write, edit, rather than "refactor code"). These three can be combined into requests nobody
planned for, such as "remind me 3 months after my last dentist visit on March 3".

The course vs. this script:
- The course describes the requests and shows no code. The tools are lesson 28's code;
  the loop runs each request as its own conversation.

https://academy.claude.com/courses/building-with-the-claude-api/agents-and-tools

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import calendar
import json
from datetime import datetime, timedelta
from anthropic.types import Message, ToolParam
from helpers.chat_helper import add_user_message, add_assistant_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_turn, log_tool_request, log_tool_response


# The tools, from lesson 28.
def get_current_datetime(date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time, formatted with date_format (strftime codes)"""
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)


def add_duration_to_datetime(datetime_str: str, duration: float = 0, unit: str = "days",
                             input_format: str = "%Y-%m-%d") -> str:
    """Adds a duration to a date and returns the new date, e.g. "Monday, June 27, 2050 12:00:00 AM" """
    if not datetime_str:
        raise ValueError("datetime_str cannot be empty")
    date = datetime.strptime(datetime_str, input_format)
    if unit in ("seconds", "minutes", "hours", "days", "weeks"):
        new_date = date + timedelta(**{unit: duration})
    elif unit == "months":
        # Months have different lengths: keep the day, capped at the last day of the target month.
        month = date.month - 1 + int(duration)
        year = date.year + month // 12
        month = month % 12 + 1
        day = min(date.day, calendar.monthrange(year, month)[1])
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        year = date.year + int(duration)
        day = min(date.day, calendar.monthrange(year, date.month)[1])
        new_date = date.replace(year=year, day=day)
    else:
        raise ValueError(f"Unsupported unit: {unit}. "
                         "Use seconds, minutes, hours, days, weeks, months or years.")
    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def set_reminder(content: str, timestamp: str) -> str:
    """Sets a reminder (here it only prints it; a real app would store it)"""
    if not content or not timestamp:
        raise ValueError("content and timestamp cannot be empty")
    print(f"----\nSetting the following reminder for {timestamp}:\n{content}\n----")
    return f"Reminder set for {timestamp}: {content}"


tool_schemas = [
    ToolParam({
        "name": "get_current_datetime",
        "description": "Returns the current date and time formatted according to the specified format",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": "A string specifying the format of the returned datetime. "
                                   "Uses Python's strftime format codes.",
                    "default": "%Y-%m-%d %H:%M:%S",
                }
            },
            "required": [],
        },
    }),
    ToolParam({
        "name": "add_duration_to_datetime",
        "description": "Adds a duration to a date and time and returns the resulting date and time. Use it "
                       "for date arithmetic instead of calculating dates yourself. Returns a readable "
                       "string, e.g. 'Monday, June 27, 2050 12:00:00 AM'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "datetime_str": {"type": "string",
                                 "description": "The starting date, in input_format (default YYYY-MM-DD)."},
                "duration": {"type": "number",
                             "description": "How much to add, in the given unit (negative: back in time)."},
                "unit": {"type": "string",
                         "enum": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"],
                         "description": "The unit of the duration."},
                "input_format": {"type": "string",
                                 "description": "The format of datetime_str, in strftime codes. "
                                                "Default '%Y-%m-%d'."},
            },
            "required": ["datetime_str"],
        },
    }),
    ToolParam({
        "name": "set_reminder",
        "description": "Sets a reminder that will notify the user at the given date and time. Work out the "
                       "exact date and time first. Returns a confirmation of the reminder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What the reminder is about."},
                "timestamp": {"type": "string",
                              "description": "When it fires, in ISO 8601, e.g. '2050-06-27T07:00:00'."},
            },
            "required": ["content", "timestamp"],
        },
    }),
]

tool_functions = {
    "get_current_datetime": get_current_datetime,
    "add_duration_to_datetime": add_duration_to_datetime,
    "set_reminder": set_reminder,
}

model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000

# Four goals for the same agent: the code below does not change between them.
requests = [
    "What time is it?",
    "What day of the week will it be in 11 days?",
    "Set a reminder for my gym session next Wednesday at 7am.",
    "When does my 1-year warranty expire?",
]


def run_tools(message: Message, tools_used: list[str]) -> list[dict]:
    """Runs every tool Claude asked for; returns one tool_result block per request"""
    results = []
    for block in [block for block in message.content if block.type == "tool_use"]:
        log_tool_request(block.name, block.input)
        tools_used.append(block.name)
        try:
            output, is_error = json.dumps(tool_functions[block.name](**block.input)), False
        except Exception as e:
            # Claude still gets an answer, and can retry with other arguments.
            output, is_error = f"Error: {e}", True
        log_tool_response(block.name, output)
        results.append({"type": "tool_result", "tool_use_id": block.id,
                        "content": output, "is_error": is_error})
    return results


def run_agent(goal: str) -> tuple[str, list[str], int]:
    """The agent: Claude gets the goal and the tools, and decides the steps.
    Returns (answer, tools used in order, number of turns)"""
    messages = []
    add_user_message(messages, goal)
    tools_used = []
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = send(messages, model=model, max_tokens=max_tokens, tools=tool_schemas)
        add_assistant_message(messages, response)
        if response.stop_reason != "tool_use":                               # <- new
            # Done, or Claude needs something only the user can give (e.g. a purchase date).
            return text_from_message(response), tools_used, turn
        add_user_message(messages, run_tools(response, tools_used))


def main():
    results = []
    for goal in requests:
        log_banner(f"GOAL: {goal}")
        results.append((goal, *run_agent(goal)))                              # <- new

    print("\n=== WHAT THE AGENT DID ===")
    for goal, _, tools_used, turns in results:
        steps = " -> ".join(tools_used) if tools_used else "no tools"
        print(f"  {goal}\n      {turns} turn(s): {steps}")

    print("\n=== ANSWER TO USER ===")
    for goal, answer, _, _ in results:
        print(f"\n--- {goal} ---\n{answer}")


if __name__ == "__main__":
    main()
