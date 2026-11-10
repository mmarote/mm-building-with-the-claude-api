"""Building with the Claude API: Lesson 8a - Structured data (prefill)

We often need Claude's answer as data (JSON) that a program can read, not as chat text.

Step 1, the default answer: JSON inside a Markdown code fence, plus a sentence of explanation.
Awkward to use in an app.

Step 2, prefill and a stop sequence:

    assistant prefill:  "```json"          we write the start of Claude's reply
    Claude continues:   { ...the JSON... }
    stop sequence:      "```"              generation stops here: no commentary after it

Then the text is parsed with json.loads().

DEPRECATED: assistant prefill is not supported on current models.
- Step 2 returns a 400 error on models such as Claude Haiku 5.5.
- The code is kept for reference, as in the temperature lesson.
- To test it anyway, point the model at Claude Haiku 4.5 (`claude-haiku-4-5`), which accepts
  prefill. Do not use this in new work.
- Lesson 8b shows the current way: structured outputs.

https://academy.claude.com/courses/building-with-the-claude-api/structured-data

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import json
import anthropic
from helpers.chat_helper import add_assistant_message, add_user_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000
user_prompt = "Generate a very short event bridge rule as json"


def chat(messages, stop_sequences=None):
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    log_model_request(model=model, max_tokens=max_tokens, messages=messages)

    response = client.messages.create(**params)

    log_model_response(response)

    return next(block.text for block in response.content if block.type == "text")


def main():
    # Step 1: the default answer, wrapped in a markdown code fence with commentary.
    messages = []
    add_user_message(messages, user_prompt)
    log_turn(1, note="default response")
    text = chat(messages)
    print(f"\n=== ANSWER TO USER (default) ===\n{text}")

    # Step 2: prefill the reply with the opening fence, and stop at the closing fence.
    messages = []
    add_user_message(messages, user_prompt)
    add_assistant_message(messages, "```json")
    log_turn(2, note="prefill and stop sequence")
    text = chat(messages, stop_sequences=["```"])

    # Claude's reply is the rest of the JSON. Strip the extra newlines before parsing.
    rule = json.loads(text.strip())
    print(f"\n=== ANSWER TO USER (prefill) ===\n{json.dumps(rule, indent=2)}")


if __name__ == "__main__":
    main()
