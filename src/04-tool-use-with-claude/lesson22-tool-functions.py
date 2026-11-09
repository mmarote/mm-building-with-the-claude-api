"""Building with the Claude API: Lesson 22 - Tool functions

A tool function is a plain Python function that our code runs when Claude asks for it; here we write the first one, get_current_datetime.

Where it fits in tool use:

    Claude: "I need the current date and time"  ->  our code runs get_current_datetime()  ->  result back to Claude
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                    this lesson

The reminder project's three tools (lesson 21):
- get_current_datetime      <- this lesson
- add_duration_to_datetime  (later)
- set_reminder              (later)

How to write a tool function:
- Clear names for the function and its arguments, so their purpose is obvious.
- Validate the arguments: reject empty or invalid values.
- Raise errors with clear messages. Claude sees the message and can retry with corrected arguments.

No Claude API call yet: this lesson only writes and tries the function. Lesson 23 describes it to
Claude with a JSON schema.

https://academy.claude.com/courses/building-with-the-claude-api/tool-functions
"""

from datetime import datetime


def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time, formatted with date_format (strftime codes)"""

    if not date_format:
        raise ValueError("date_format cannot be empty")
    
    return datetime.now().strftime(date_format)


def main():
    # The default format, e.g. "2024-01-15 14:30:25"
    print(f'get_current_datetime()             -> {get_current_datetime()}')

    # Just hour and minute, e.g. "14:30"
    print(f'get_current_datetime("%H:%M")      -> {get_current_datetime("%H:%M")}')

    # An invalid argument: the error message is what Claude would see.
    try:
        get_current_datetime("")
    except ValueError as error:
        print(f'get_current_datetime("")           -> ValueError: {error}')


if __name__ == "__main__":
    main()
