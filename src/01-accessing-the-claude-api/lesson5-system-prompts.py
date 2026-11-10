"""Building with the Claude API: Lesson 5 - System prompts

The same question, asked twice: once with no instructions, and once with a system prompt
that turns Claude into a math tutor.

    Turn 1:  question                           ->  Claude answers it directly
    Turn 2:  question + system prompt (a tutor) ->  Claude guides step by step, without giving the answer

What a system prompt is:
- It sets Claude's role and style for the whole conversation.
- It goes in the system parameter, separate from the messages.

https://academy.claude.com/courses/building-with-the-claude-api/system-prompts

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import anthropic
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000
user_prompt = "Solve for x: 3x + 7 = 22"
messages = [{"role": "user", "content": user_prompt}]

system_prompt = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step."""

def chat(messages, system=None):
    # The API does not accept system=None, so the parameter is only added when there is a system prompt.
    params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages
        }
    
    if system:
        params["system"] = system

    log_model_request(model=model, max_tokens=max_tokens, system=system, messages=messages)

    response = client.messages.create(**params)

    log_model_response(response)
    return response.content[0].text


def main():
    # Turn 1: no system prompt. Claude answers the way it normally would.
    log_turn(1, note="no system prompt")
    answer = chat(messages)
    print(f"\n=== ANSWER TO USER ===\n{answer}")

    # Turn 2: the same question, now with the tutor system prompt.
    log_turn(2, note="with the tutor system prompt")
    answer = chat(messages, system=system_prompt)
    print(f"\n=== ANSWER TO USER ===\n{answer}")


if __name__ == "__main__":
    main()
