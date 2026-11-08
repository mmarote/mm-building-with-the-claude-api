"""Building with the Claude API: Lesson 3 - Making a request

Send your first request to the Anthropic API: one user message in, one answer out.

    your code  --- messages.create(model, max_tokens, messages) --->  Claude
               <--- response (answer in response.content[0].text) ---

The three required parameters of client.messages.create():
- model:      which Claude model answers
- max_tokens: the most tokens the answer may use
- messages:   the conversation so far (here, a single user message)

https://academy.claude.com/courses/building-with-the-claude-api/making-a-request

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import anthropic
from helpers.models import CHEAPEST
from helpers.log_helper import log_model_request, log_model_response, log_answer_to_user


client = anthropic.Anthropic()

model = CHEAPEST
# A safety limit on the response length, not a target: Claude stops earlier when its answer is complete.
max_tokens = 1000
user_prompt = "What is quantum computing? Answer in one sentence"
messages = [{"role": "user", "content": user_prompt}]


def main():

    log_model_request(model=model, max_tokens=max_tokens, messages=messages)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )

    log_model_response(response)
    log_answer_to_user(response)


if __name__ == "__main__":
    main()
