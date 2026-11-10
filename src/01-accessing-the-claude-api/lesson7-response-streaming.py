"""Building with the Claude API: Lesson 7 - Response streaming

With streaming, Claude sends the answer in small pieces while it writes, so the user sees
text appear right away.

    normal:     [ wait ............................ ]  whole answer
    streaming:  piece, piece, piece, piece, ...        (text appears as it is written)

- Part 1: the raw events that come back.
- Part 2: the SDK helper that gives only the text as it arrives, then the complete message
  at the end.

https://academy.claude.com/courses/building-with-the-claude-api/response-streaming

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import anthropic
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn
from helpers.chat_helper import add_user_message


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000
messages = []


def main():
    add_user_message(messages, "Write a 1 sentence description of a fake database")

    # Part 1: raw events. Each event says what part of the message it is (start, text chunk, end).
    log_turn(1, note="Basic streaming: raw events")
    stream = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        stream=True,    # NEW: enabling basic streaming
    )
    for event in stream:
        print(f"raw event: {event}")

    # Part 2: the SDK helper. text_stream yields only the text chunks, as they arrive.
    log_turn(2, note="Simplified text streaming: text stream")
    print("\n=== ANSWER TO USER (streamed) ===")
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(f"text event: {text}")
        print()
        # After the loop, the complete message is still available, for example to save it.
        final_message = stream.get_final_message()

    print(f"\nFinal message: {final_message}")
    print(f"\nTokens used: input {final_message.usage.input_tokens}, output {final_message.usage.output_tokens}")


if __name__ == "__main__":
    main()
