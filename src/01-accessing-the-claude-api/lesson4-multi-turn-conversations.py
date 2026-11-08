"""Building with the Claude API: Lesson 4 - Multi-turn conversations

Claude does not remember earlier requests, so our code keeps the conversation and sends all of it every time.

The messages list grows with each turn, and the whole list goes with every request:

    request 1:  [user]                       "Define quantum computing in one sentence"
    request 2:  [user, assistant, user]      ... + Claude's answer + "Write another sentence"

- user:      what we ask
- assistant: Claude's reply, added to the list before the next question

https://academy.claude.com/courses/building-with-the-claude-api/multi-turn-conversations

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import anthropic
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def chat(messages):
    log_model_request(model=model, max_tokens=max_tokens, messages=messages)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    log_model_response(response)
    return response.content[0].text


def main():
    messages = []

    # Turn 1: the first question. Claude has no earlier history to look at.
    log_turn(1)
    add_user_message(messages, "Define quantum computing in one sentence")
    answer = chat(messages)
    add_assistant_message(messages, answer)

    # Turn 2: a follow-up. The request now carries the whole conversation, so Claude knows what "another" refers to.
    log_turn(2)
    add_user_message(messages, "Write another sentence")
    answer = chat(messages)
    add_assistant_message(messages, answer)

    print(f"\n=== ANSWER TO USER ===\n{answer}")


if __name__ == "__main__":
    main()
