"""Building with the Claude API: Lesson 39 - Extended thinking

With extended thinking, Claude reasons before it answers, and the reply has two kinds
of blocks: thinking (the reasoning) and text (the answer).

The reply:

    content: [ thinking block ──► the reasoning (or a summary of it) + a signature
               text block     ──► the answer ]

- signature: proves the thinking was not changed. Send thinking blocks back exactly as received.
- redacted_thinking: thinking flagged by safety systems comes back encrypted. Pass it back
  unchanged too; you cannot read it.

When to use it:
- First get the prompt right without thinking (lessons 15-19).
- Turn thinking on only if accuracy is still not good enough: thinking tokens are billed as
  output, and the answer takes longer.

The course vs. Haiku 5.5 (the course uses an older model):

    course (older models)                       Haiku 5.5 (this script)
    thinking={"type": "enabled",                thinking={"type": "adaptive",
              "budget_tokens": 1024}                      "display": "summarized"}
    you set a token budget (min 1024)           Claude decides how much to think;
                                                budget_tokens is rejected (400)
    off unless you ask                          on by default
    thinking text returned                      thinking text empty unless display="summarized"

- To make Claude think more or less, use effort: output_config={"effort": "low" ... "max"}
  (default "medium"). Not used here.
- Thinking counts toward max_tokens: too low a limit leaves no room for the answer.
- No assistant prefill and no temperature with thinking (or at all, on Haiku 5.5).

https://academy.claude.com/courses/building-with-the-claude-api/extended-thinking

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import anthropic
from helpers.chat_helper import add_user_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_model_request, log_model_response, log_model_usage


client = anthropic.Anthropic()

model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000
# "summarized": we get a summary of the reasoning. Without display, the thinking block is empty
# (only its signature), but Claude still thinks.
thinking = {"type": "adaptive", "display": "summarized"}
user_prompt = "Write a short guide on recursion."


def main():
    messages = []
    add_user_message(messages, user_prompt)

    log_model_request(model=model, max_tokens=max_tokens, thinking=thinking, messages=messages)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking=thinking,
        messages=messages,
    )
    log_model_response(response)
    log_model_usage(response)

    # Read the blocks by type: the reply can start with one or more thinking blocks.
    for block in response.content:
        if block.type == "thinking":
            print(f"\n=== THINKING (summary) ===\n{block.thinking or '(empty: no display requested)'}")
            print(f"\nsignature: {block.signature[:40]}... ({len(block.signature)} characters)")
        elif block.type == "redacted_thinking":
            print(f"\n=== REDACTED THINKING ===\nencrypted, {len(block.data)} characters: pass it back unchanged")

    answer = "".join(block.text for block in response.content if block.type == "text")
    print(f"\n=== ANSWER TO USER ===\n{answer}")


if __name__ == "__main__":
    main()
