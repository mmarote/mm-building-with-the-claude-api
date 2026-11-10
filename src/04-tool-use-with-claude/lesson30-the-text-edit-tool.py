"""Building with the Claude API: Lesson 30 - The text edit tool

The text editor is a tool built into Claude: we send only a short stub, Claude already knows
the full schema, and our code carries out the file operations it asks for.

What we send (Claude expands the stub into the full tool definition):

    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}

What Claude can ask for (tool_use.input["command"]), and what our code does:

    view         path, view_range (optional)    show a file (with line numbers) or list a folder
    create       path, file_text                write a new file (a backup is kept if it existed)
    str_replace  path, old_str, new_str         replace one exact piece of text (error if 0 or 2+ matches)
    insert       path, insert_line, insert_text insert text after a line (0 = at the start)

Safety: path is untrusted model output.
- Every file operation is confined to one workspace folder:
  data/04-tool-use-with-claude/text-editor-workspace/
- A path that leads outside it (.., an absolute path, a symlink) is rejected with an error,
  which goes back to Claude like any other tool error.

The rest is the tool loop of lesson 27: run_tool sends each command to the matching function.

First run: the workspace is created with a small main.py, so the course's prompt has a file to open.
Delete the folder to start again.

Version strings: the course's get_text_edit_schema picks the version by model. Claude 3 models use
text_editor_20250124 ("str_replace_editor"); current models use text_editor_20250728.

https://academy.claude.com/courses/building-with-the-claude-api/the-text-edit-tool

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import shutil
from pathlib import Path
from helpers.chat_helper import add_user_message, add_assistant_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_tool_request, log_tool_response


model = CHEAPEST
# Claude thinks before it answers, and thinking counts toward max_tokens. Writing files takes room:
# too low a limit cuts a file off. You only pay for the tokens Claude actually uses.
max_tokens = 16000
user_prompt = ("Open the ./main.py file and write out a function to calculate pi to the 5th digit. "
               "Then create a ./test.py file to test your implementation.")

# All file operations happen inside this folder (in the data folder at the project root).
WORKSPACE = (Path(__file__).resolve().parents[2] / "data" / "04-tool-use-with-claude" / "text-editor-workspace").resolve()

STARTER_MAIN = '''"""A small starter program for the text editor lesson."""


def greet(name):
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("world"))
'''


def get_text_edit_schema(model):
    """The text editor stub for this model (from the course)"""
    if model.startswith("claude-3"):
        return {"type": "text_editor_20250124", "name": "str_replace_editor"}
    else:
        return {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}


# ── The file operations Claude can ask for ──────────────────────────────────

def safe_path(path):
    """Resolves Claude's path inside the workspace; refuses anything that leads outside it"""
    target = (WORKSPACE / path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"Path {path!r} is outside the workspace. Use a relative path such as ./main.py")
    return target


def view(path, view_range=None):
    target = safe_path(path)
    if target.is_dir():
        entries = sorted(p.relative_to(WORKSPACE).as_posix() + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(entries) or "(empty folder)"
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lines = target.read_text(encoding="utf-8").splitlines()
    first, last = 1, len(lines)
    if view_range:
        first = view_range[0]
        last = len(lines) if view_range[1] == -1 else view_range[1]
    # Line numbers help Claude with insert and with picking the right text to replace.
    return "\n".join(f"{number:6}\t{lines[number - 1]}" for number in range(first, last + 1))


def create(path, file_text):
    target = safe_path(path)
    if target.exists():
        shutil.copy(target, target.with_name(target.name + ".bak"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(file_text, encoding="utf-8")
    return f"File created: {path}"


def str_replace(path, old_str, new_str=""):
    target = safe_path(path)
    text = target.read_text(encoding="utf-8")
    matches = text.count(old_str)
    if matches == 0:
        raise ValueError(f"old_str not found in {path}")
    if matches > 1:
        raise ValueError(f"old_str appears {matches} times in {path}; include more context so it is unique")
    target.write_text(text.replace(old_str, new_str), encoding="utf-8")
    return f"Text replaced in {path}"


def insert(path, insert_line, insert_text):
    target = safe_path(path)
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    if not 0 <= insert_line <= len(lines):
        raise ValueError(f"insert_line must be between 0 and {len(lines)}")
    lines.insert(insert_line, insert_text if insert_text.endswith("\n") else insert_text + "\n")
    target.write_text("".join(lines), encoding="utf-8")
    return f"Text inserted in {path} after line {insert_line}"


# ── The tool loop of lesson 27 ──────────────────────────────────────────────

def run_tool(tool_name, tool_input):
    if tool_name != "str_replace_based_edit_tool":
        raise ValueError(f"Unknown tool: {tool_name}")

    command = tool_input["command"]
    arguments = {key: value for key, value in tool_input.items() if key != "command"}
    if command == "view":
        return view(**arguments)
    elif command == "create":
        return create(**arguments)
    elif command == "str_replace":
        return str_replace(**arguments)
    elif command == "insert":
        return insert(**arguments)
    raise ValueError(f"Unknown command: {command}")


def run_tools(message):
    tool_result_blocks = []
    for tool_request in [block for block in message.content if block.type == "tool_use"]:
        log_tool_request(tool_request.name, tool_request.input)
        try:
            content, is_error = run_tool(tool_request.name, tool_request.input), False
        except Exception as e:
            # Claude sees the error and can try again (e.g. with a different path or old_str).
            content, is_error = f"Error: {e}", True
        log_tool_response(tool_request.name, content)
        tool_result_blocks.append({
            "type": "tool_result",
            "tool_use_id": tool_request.id,
            "content": content,
            "is_error": is_error,
        })
    return tool_result_blocks


def run_conversation(messages):
    turn = 0
    while True:
        turn += 1
        log_turn(turn)
        response = send(messages, model=model, max_tokens=max_tokens, tools=[get_text_edit_schema(model)])
        add_assistant_message(messages, response)

        if response.stop_reason != "tool_use":
            break

        # Claude's text between tool calls (e.g. "First I'll work out the date"). The course prints it
        # before the check above; printing it here shows the final answer only once, in main().
        print(text_from_message(response))

        add_user_message(messages, run_tools(response))

    return messages


def main():
    # A starter main.py, so the prompt has a file to open.
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not (WORKSPACE / "main.py").exists():
        (WORKSPACE / "main.py").write_text(STARTER_MAIN, encoding="utf-8")

    messages = []
    add_user_message(messages, user_prompt)

    messages = run_conversation(messages)

    # The last message is Claude's final answer.
    print(f"\n=== ANSWER TO USER ===\n{text_from_message(messages[-1])}")

    print(f"\nFiles in the workspace ({WORKSPACE}):")
    print(view("."))


if __name__ == "__main__":
    main()
