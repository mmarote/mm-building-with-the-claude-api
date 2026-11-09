"""Building with the Claude API: Lesson 23 - Tool schemas

A tool schema describes a tool function to Claude: its name, what it does, and its arguments; here we write the schema for get_current_datetime.

Where it fits in tool use:

    our code sends: question + tool schemas  ->  Claude decides to use a tool  ->  our code runs the function
                               ^^^^^^^^^^^^
                               this lesson (the function itself is from lesson 22)

The parts of a schema:

    name          "get_current_datetime"  must match the function, so our code knows what to run
    description   what the tool does, when to use it, what it returns
    input_schema  the arguments, as JSON Schema: type, description, default, and which are required

How to write a good schema:
- A description of about 3-4 sentences: what the tool does, when Claude should use it, what it returns.
- Describe every argument in detail.
- Name the variable <function name>_schema, so it is clear which function it belongs to.
- Shortcut: give Claude the function's code and Anthropic's tool use docs, and ask it to write the
  schema ("Write a valid JSON schema spec for the purposes of tool calling for this function").

ToolParam (from anthropic.types) is optional: it only adds type checking in the editor.

No Claude API call yet: lesson 24 sends the schema to Claude.

https://academy.claude.com/courses/building-with-the-claude-api/tool-schemas
"""

import json
from datetime import datetime
from anthropic.types import ToolParam


# The tool function, from lesson 22.
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


def main():
    print("Tool schema (what Claude will see):")
    print(json.dumps(get_current_datetime_schema, indent=2))

    # The schema and the function must agree, or Claude would ask for something our code can't run.
    print(f"\nname matches the function:    {get_current_datetime_schema['name'] == get_current_datetime.__name__}")
    default = get_current_datetime_schema["input_schema"]["properties"]["date_format"]["default"]
    print(f"default matches the function: {default == get_current_datetime.__defaults__[0]}")


if __name__ == "__main__":
    main()
