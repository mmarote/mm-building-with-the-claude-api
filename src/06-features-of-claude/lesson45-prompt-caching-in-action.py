"""Building with the Claude API: Lesson 45 - Prompt caching in action

Prompt caching lets Claude reuse the work on a long, unchanged start of a request;
here we mark the tools and the system prompt for caching and watch the usage counters
over three requests.

How a request is cached: in this order, up to each cache_control marker.

    tools ──► system prompt ──► messages
          ▲                 ▲
          marker on the     marker on the system block
          LAST tool         (the system prompt must be a list of blocks, not a plain string)

    {"type": "ephemeral"}: kept 5 minutes, and the 5 minutes restart on each use
    (a 1-hour cache is possible, at a higher price).

What to watch in the usage:

    cache_creation_input_tokens   written to the cache   (a bit more than normal input)
    cache_read_input_tokens       read from the cache    (about a tenth of normal input)
    input_tokens                  the rest, not cached

Our three requests, same tools and system prompt except the third:

    1. first question                    expect: cache written (creation > 0)
    2. a different question              expect: cache read    (read > 0): only the question is new
    3. one word changed in the system    expect: the system prompt (and what follows) is written again

Rules (lesson 44):
- Any change to a cached part invalidates that part and everything after it.
- A prefix shorter than the model's minimum is not cached, without an error (only zeros in usage).
  Haiku 5.5's minimum is 512 tokens; our system prompt holds the whole report to pass it.

The course's prompt is a 6K-token coding assistant; ours is an analyst assistant with the
Northwind report (lessons 33-38) and two tool schemas from lesson 28.

https://academy.claude.com/courses/building-with-the-claude-api/prompt-caching-in-action

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from pathlib import Path
from helpers.chat_helper import add_user_message, send
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_usage, Stopwatch


report_file = Path(__file__).resolve().parents[2] / "data" / "05-rag-and-agentic-search" / "report.md"

model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000

system_prompt = f"""You are an analyst assistant for Northwind Research Group. You answer questions from staff
about the group's annual report, which is included below.

Guidelines:
- Answer only from the report. If the report does not contain the answer, say so plainly.
- Quote exact figures (percentages, amounts, dates) as they appear in the report.
- Name the section your answer comes from, e.g. "(Section 3: Financial Analysis)".
- Keep answers to three sentences or fewer unless the question asks for more detail.
- If a question mixes several topics, answer each topic in its own short paragraph.
- Do not speculate about figures for years the report does not cover.
- Use plain language: the readers are staff from every division, not only specialists.

<report>
{report_file.read_text(encoding="utf-8")}
</report>"""

# Two tool schemas from lesson 28. They are here to show how tools are cached; the questions
# below do not need them.
tools = [
    {
        "name": "add_duration_to_datetime",
        "description": "Adds a duration to a datetime string and returns the resulting datetime, formatted as a readable string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "datetime_str": {"type": "string", "description": "The starting date, in the format given by input_format."},
                "duration": {"type": "number", "description": "How much time to add (can be negative)."},
                "unit": {"type": "string", "enum": ["seconds", "minutes", "hours", "days", "weeks", "months", "years"],
                         "description": "The unit of duration."},
                "input_format": {"type": "string", "description": "The strftime format of datetime_str. Default: %Y-%m-%d."},
            },
            "required": ["datetime_str"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Sets a reminder for the user at the given time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What the reminder is about."},
                "timestamp": {"type": "string", "description": "When the reminder should go off, as a readable date and time."},
            },
            "required": ["content", "timestamp"],
        },
    },
]


def with_cache_markers(tools, system):
    """Returns the tools and the system prompt with a cache_control marker (from the course)"""
    # Tools: the marker goes on the last tool. Copies, so the original list stays unchanged.
    tools_clone = tools.copy()
    last_tool = tools_clone[-1].copy()
    last_tool["cache_control"] = {"type": "ephemeral"}      # <- new
    tools_clone[-1] = last_tool

    # System prompt: a plain string has no place for a marker, so it becomes a text block.
    system_blocks = [                                       # <- new: a list of blocks, not a string
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},         # <- new
        }
    ]
    return tools_clone, system_blocks


def ask(question, system):
    messages = []
    add_user_message(messages, question)
    cached_tools, cached_system = with_cache_markers(tools, system)
    with Stopwatch() as sw:
        response = send(messages, model=model, max_tokens=max_tokens, system=cached_system, tools=cached_tools)
    log_model_usage(response, elapsed=sw.seconds)
    return response


def what_happened(usage):
    """One line on what the cache did in a request, from its usage counters"""
    written = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    if written and read:
        return "part read, rest written again"
    if read:
        return "read from the cache"
    if written:
        return "written to the cache"
    return "not cached (below the minimum?)"


def print_cache_summary(rows):
    """A plain-text table (readable in the terminal and in the runner UI)"""
    header = ("request", "written", "read", "uncached", "output", "what happened")
    lines = [
        (label, usage.cache_creation_input_tokens or 0, usage.cache_read_input_tokens or 0,
         usage.input_tokens, usage.output_tokens, what_happened(usage))
        for label, usage in rows
    ]
    widths = [max(len(str(row[i])) for row in [header] + lines) for i in range(len(header))]
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row_text(row):
        cells = []
        for i, value in enumerate(row):
            # Numbers right-aligned, text left-aligned.
            cell = str(value).rjust(widths[i]) if isinstance(value, int) else str(value).ljust(widths[i])
            cells.append(f" {cell} ")
        return "|" + "|".join(cells) + "|"

    print("\n=== CACHE SUMMARY ===")
    print(border)
    print(row_text(header))
    print(border)
    for line in lines:
        print(row_text(line))
    print(border)
    print("  written  = cache_creation_input_tokens   (a bit more than normal input)")
    print("  read     = cache_read_input_tokens       (about a tenth of normal input)")
    print("  uncached = input_tokens                  (normal price)")


def main():
    requests = [
        ("1. FIRST QUESTION (expect: cache written)", "1. first question",
         "How much did revenue grow, and why?", system_prompt),
        ("2. ANOTHER QUESTION, SAME TOOLS AND SYSTEM (expect: cache read)", "2. another question",
         "When does the third phase of the XR-472 trial start?", system_prompt),
        ("3. ONE WORD CHANGED IN THE SYSTEM PROMPT (expect: system written again)",
         "3. system prompt changed",
         "When does the third phase of the XR-472 trial start?",
         system_prompt.replace("three sentences", "two sentences")),
    ]

    rows = []
    for title, label, question, system in requests:
        log_banner(title)
        response = ask(question, system)
        rows.append((label, response.usage))

    print_cache_summary(rows)


if __name__ == "__main__":
    main()
