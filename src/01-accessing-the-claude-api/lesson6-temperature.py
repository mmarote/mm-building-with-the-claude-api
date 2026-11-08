"""Building with the Claude API: Lesson 6 - Temperature

Temperature (0.0 to 1.0) controls how predictable Claude's answers are.

Claude picks each next word from a set of likely candidates:

    0.0  ->  almost always the most likely word  ->  the answers repeat
    1.0  ->  picks more freely                   ->  the answers vary

Here the same request is sent 3 times at 0.0 and 3 times at 1.0, so you can see the difference.

DEPRECATED: this lesson no longer works on current models.
- Models released after Claude Opus 4.6 (Opus 4.7/4.8, all 5.x models, Haiku 5.5) accept only the
  default temperature, 1.0. Any other value returns a 400.
- From API version 2026-10-21, sending temperature returns a 400 on every model.
- The code is kept for reference and will fail with a 400 error.
- To test it anyway, point the model at Claude Haiku 4.5 (`claude-haiku-4-5`), which still accepts
  temperature. Do not use this in new work.

https://academy.claude.com/courses/building-with-the-claude-api/temperature

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import anthropic
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000
user_prompt = "Suggest one movie idea in one sentence."
messages = [{"role": "user", "content": user_prompt}]


def chat(messages, temperature=1.0):
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,  # Deprecated: rejected by current models, see the note at the top.
    }

    log_model_request(model=model, max_tokens=max_tokens, messages=messages)

    response = client.messages.create(**params)

    log_model_response(response)
    return response.content[0].text


def main():
    for turn in range(1, 4):
        log_turn(turn, note="temperature 0.0")
        answer = chat(messages, temperature=0.0)
        print(f"\n=== ANSWER TO USER ===\n{answer}")

    for turn in range(4, 7):
        log_turn(turn, note="temperature 1.0")
        answer = chat(messages, temperature=1.0)
        print(f"\n=== ANSWER TO USER ===\n{answer}")


if __name__ == "__main__":
    main()
