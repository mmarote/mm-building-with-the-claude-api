"""Building with the Claude API: Lesson 29 - Fine grained tool calling

With streaming, a tool's arguments arrive in pieces; fine-grained tool calling sends each
piece as soon as Claude writes it, instead of in validated bursts.

Without fine-grained tool calling (the default):

    Claude writes:   {"abstract": "This paper ...", "meta": {"word_count": 847, "review": "..."}}
    API sends:       [wait] whole "abstract" value  [wait] whole "meta" object
                     the API buffers each top-level key, validates it, then sends it at once

With fine-grained tool calling:

    API sends:       {"abs  tract": "This  paper ...  "meta": {"word_  count": 847 ...
                     every piece right away: word_count arrives before "meta" is finished

How to turn it on:
- Add "eager_input_streaming": True to each tool definition (see with_fine_grained below).
- Stream the response (client.messages.stream). The "input_json" events carry the pieces:
  event.partial_json is the new piece, event.snapshot is everything parsed so far.

The catch: the API no longer validates the tool input, so the code must:
- Handle JSON that cannot be parsed at all: the stream raises ValueError (retry the request).
- Check the parsed input before running the tool: it can be incomplete or the wrong shape
  (e.g. "word_count": undefined).
Use it only when real-time feedback matters and the code handles bad JSON; otherwise the default
(validated) behaviour is fine.

Set fine_grained = False below and run again to compare: fewer, larger pieces, and a longer wait
before the first one.

Different from the course: the course turns this on with a beta header (fine_grained=True in its
run_conversation). Today it is the eager_input_streaming field on the tool, with no beta header. In
this SDK, event.snapshot is already parsed, so there is no json.loads(event.snapshot) step.

https://academy.claude.com/courses/building-with-the-claude-api/fine-grained-tool-calling

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import json
import time
import anthropic
from anthropic.types import ToolParam
from helpers.chat_helper import add_user_message, add_assistant_message, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response, log_tool_request, log_tool_response


# The tool: Claude saves an article summary with a nested "meta" object.
def save_article(abstract, meta):
    """Saves the article's abstract and metadata (here it only prints them; a real app would store them)"""
    print(f"\n----\nSaving article:\n  abstract:   {abstract}\n  word_count: {meta['word_count']}\n  review:     {meta['review']}\n----")
    return "Article saved"


save_article_schema = ToolParam({
    "name": "save_article",
    "description": "Saves an article's abstract together with metadata about the article. Use it when the user "
                   "asks to save an article. Returns a confirmation once the article is saved.",
    "input_schema": {
        "type": "object",
        "properties": {
            "abstract": {"type": "string", "description": "A short abstract of the article, 2-3 sentences."},
            "meta": {
                "type": "object",
                "properties": {
                    "word_count": {"type": "integer", "description": "The number of words in the article."},
                    "review": {"type": "string", "description": "A short review of the article, 2-3 sentences."},
                },
                "required": ["word_count", "review"],
            },
        },
        "required": ["abstract", "meta"],
    },
})


client = anthropic.Anthropic()

model = CHEAPEST
# Claude thinks before it answers, and thinking counts toward max_tokens; too low a limit can leave
# no answer at all. You only pay for the tokens Claude actually uses.
max_tokens = 16000
# Turn fine-grained tool calling on or off, to compare.
fine_grained = True

article = """QuanNet: Quantum-Inspired Neural Networks for Small Datasets.
We introduce QuanNet, a neural network architecture whose layers borrow the idea of superposition
from quantum computing: each neuron keeps several candidate weights and combines them during
training. On five small image datasets, QuanNet matches larger models while using 40% fewer
parameters, and it overfits less when fewer than 1,000 training examples are available. We also
show that its training time grows linearly with the number of candidate weights, which keeps it
practical on ordinary hardware. The code and datasets are publicly available."""

user_prompt = f"Save this article with an abstract and a short review:\n\n<article>\n{article}\n</article>"


def with_fine_grained(tools):
    """Copies of the tool definitions with eager_input_streaming turned on"""
    return [{**tool, "eager_input_streaming": True} for tool in tools]


def run_tool(tool_name, tool_input):
    if tool_name == "save_article":
        return save_article(**tool_input)
    raise ValueError(f"Unknown tool: {tool_name}")


def input_is_valid(tool_input):
    """Checks the shape of save_article's input: with fine-grained calling the API no longer does"""
    return (
        isinstance(tool_input, dict)
        and isinstance(tool_input.get("abstract"), str)
        and isinstance(tool_input.get("meta"), dict)
        and isinstance(tool_input["meta"].get("word_count"), int)
        and isinstance(tool_input["meta"].get("review"), str)
    )


def run_tools(message):
    tool_result_blocks = []
    for tool_request in [block for block in message.content if block.type == "tool_use"]:
        log_tool_request(tool_request.name, tool_request.input)
        if not input_is_valid(tool_request.input):
            # Tell Claude what was wrong, so it can send the arguments again.
            content, is_error = f"Error: invalid input {json.dumps(tool_request.input)}", True
        else:
            try:
                content, is_error = json.dumps(run_tool(tool_request.name, tool_request.input)), False
            except Exception as e:
                content, is_error = f"Error: {e}", True
        log_tool_response(tool_request.name, content)
        tool_result_blocks.append({
            "type": "tool_result",
            "tool_use_id": tool_request.id,
            "content": content,
            "is_error": is_error,
        })
    return tool_result_blocks


def stream_turn(messages, tools):
    """Streams one reply from Claude, printing text and tool-input pieces as they arrive"""
    log_model_request(model=model, max_tokens=max_tokens, messages=messages, tools=tools)
    print("\nStreaming (tool-input pieces separated by |):")

    start = time.perf_counter()
    first_piece_at = None
    pieces = 0

    with client.messages.stream(model=model, max_tokens=max_tokens, messages=messages, tools=tools) as stream:
        for event in stream:
            if event.type == "text":
                print(event.text, end="", flush=True)
            elif event.type == "input_json":
                # event.partial_json: the new piece. event.snapshot: everything parsed so far.
                if first_piece_at is None:
                    first_piece_at = time.perf_counter() - start
                pieces += 1
                print(event.partial_json, end="|", flush=True)
        response = stream.get_final_message()

    if pieces:
        print(f"\n\n{pieces} tool-input pieces; the first arrived after {first_piece_at:.1f}s")
    log_model_response(response)
    return response


def run_conversation(messages, tools, fine_grained=False):
    """The tool loop of lesson 27, streaming each reply; fine_grained turns on eager_input_streaming"""
    if fine_grained:
        tools = with_fine_grained(tools)

    turn = 0
    json_retries = 0
    while True:
        turn += 1
        log_turn(turn, note="fine-grained" if fine_grained else "default streaming")
        try:
            response = stream_turn(messages, tools)
        except ValueError:
            # Claude sent tool-input JSON that cannot be parsed at all. Ask again (a few times at most).
            print("\nReceived invalid JSON, sending the request again...")
            json_retries += 1
            if json_retries > 2:
                raise
            continue
        json_retries = 0

        add_assistant_message(messages, response)

        if response.stop_reason != "tool_use":
            break

        add_user_message(messages, run_tools(response))

    return messages


def main():
    messages = []
    add_user_message(messages, user_prompt)

    messages = run_conversation(messages, tools=[save_article_schema], fine_grained=fine_grained)

    print(f"\n=== ANSWER TO USER ===\n{text_from_message(messages[-1])}")


if __name__ == "__main__":
    main()
