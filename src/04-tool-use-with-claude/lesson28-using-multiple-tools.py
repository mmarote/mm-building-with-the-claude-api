"""Building with the Claude API: Lesson 28 - Using multiple tools

We add the reminder project's other two tools, so Claude can chain several tools to finish one request; the loop itself does not change.

The three tools (lesson 21's project, complete):

    get_current_datetime(date_format)                          the current date and time   (lesson 22)
    add_duration_to_datetime(datetime_str, duration, unit)     date arithmetic             (new)
    set_reminder(content, timestamp)                           sets the reminder           (new)

Adding a tool takes three steps, and the loop stays the same:
1. Write the function and its schema.
2. Add a branch for it in run_tool.
3. Add its schema to the tools list.

Example: "Set a reminder for my doctors appointment. Its 177 days after Jan 1st, 2050."

    Claude:   "First I'll work out the date"   -> add_duration_to_datetime("2050-01-01", 177, "days")
    our code: "Monday, June 27, 2050 12:00:00 AM"
    Claude:   "Now I'll set the reminder"      -> set_reminder("Doctor's appointment", "2050-06-27T00:00:00")
    our code: reminder set
    Claude:   final answer

The course provides add_duration_to_datetime and set_reminder but does not show their code; the
versions here are written to match its example output. set_reminder only prints the reminder: a
real app would store it.

The helpers come from helpers/chat_helper.py (lesson 26 writes them out in full). The course's chat()
is send() there: it returns the whole message. chat() in chat_helper returns only the text.

https://academy.claude.com/courses/building-with-the-claude-api/implementing-multiple-turns

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import json
import calendar
from datetime import datetime, timedelta
from anthropic.types import ToolParam
from helpers.chat_helper import add_user_message, add_assistant_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_tool_request, log_tool_response


# The tools. get_current_datetime is from lessons 22-23; the other two are new.
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



def add_duration_to_datetime(datetime_str, duration=0, unit="days", input_format="%Y-%m-%d"):
    """Adds a duration to a date and returns the new date, e.g. "Monday, June 27, 2050 12:00:00 AM" """
    if not datetime_str:
        raise ValueError("datetime_str cannot be empty")

    date = datetime.strptime(datetime_str, input_format)

    if unit == "seconds":
        new_date = date + timedelta(seconds=duration)
    elif unit == "minutes":
        new_date = date + timedelta(minutes=duration)
    elif unit == "hours":
        new_date = date + timedelta(hours=duration)
    elif unit == "days":
        new_date = date + timedelta(days=duration)
    elif unit == "weeks":
        new_date = date + timedelta(weeks=duration)
    elif unit == "months":
        # Months have different lengths: keep the day, capped at the last day of the target month.
        month = date.month - 1 + duration
        year = date.year + month // 12
        month = month % 12 + 1
        day = min(date.day, calendar.monthrange(year, month)[1])
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        year = date.year + duration
        day = min(date.day, calendar.monthrange(year, date.month)[1])
        new_date = date.replace(year=year, day=day)
    else:
        raise ValueError(f"Unsupported unit: {unit}. Use seconds, minutes, hours, days, weeks, months or years.")

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


add_duration_to_datetime_schema = ToolParam({
    "name": "add_duration_to_datetime",
    "description": "Adds a duration to a date and time and returns the resulting date and time. Use it for "
                   "date arithmetic, such as finding the date a number of days or months after a given date, "
                   "instead of calculating dates yourself. Returns the result as a readable string, e.g. "
                   "'Monday, June 27, 2050 12:00:00 AM'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {
                "type": "string",
                "description": "The starting date and time, written in input_format (by default YYYY-MM-DD, e.g. '2050-01-01').",
            },
            "duration": {
                "type": "number",
                "description": "How much to add, in the given unit. Use a negative number to go back in time.",
            },
            "unit": {
                "type": "string",
                "enum": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"],
                "description": "The unit of the duration.",
            },
            "input_format": {
                "type": "string",
                "description": "The format of datetime_str, in Python's strftime codes. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
})


def set_reminder(content, timestamp):
    """Sets a reminder (here it only prints it; a real app would store it)"""
    if not content:
        raise ValueError("content cannot be empty")
    if not timestamp:
        raise ValueError("timestamp cannot be empty")

    print(f"----\nSetting the following reminder for {timestamp}:\n{content}\n----")
    return f"Reminder set for {timestamp}: {content}"


set_reminder_schema = ToolParam({
    "name": "set_reminder",
    "description": "Sets a reminder that will notify the user at the given date and time. Use it when the "
                   "user asks to be reminded of something. Work out the exact date and time first (for "
                   "example with add_duration_to_datetime). Returns a confirmation of the reminder.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "What the reminder is about, e.g. 'Doctor's appointment'.",
            },
            "timestamp": {
                "type": "string",
                "description": "When the reminder fires, in ISO 8601 format, e.g. '2050-06-27T00:00:00'.",
            },
        },
        "required": ["content", "timestamp"],
    },
})


model = CHEAPEST
# Claude thinks before it answers, and thinking counts toward max_tokens; too low a limit can leave
# no answer at all. You only pay for the tokens Claude actually uses.
max_tokens = 16000
user_prompt = "Set a reminder for my doctors appointment. Its 177 days after Jan 1st, 2050."


def run_tool(tool_name, tool_input):
    """Runs the tool function with this name; add a branch here for every new tool"""
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "add_duration_to_datetime":
        return add_duration_to_datetime(**tool_input)
    elif tool_name == "set_reminder":
        return set_reminder(**tool_input)
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
        response = send(messages, model=model, max_tokens=max_tokens, tools=[
            get_current_datetime_schema,
            add_duration_to_datetime_schema,
            set_reminder_schema,
        ])
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
