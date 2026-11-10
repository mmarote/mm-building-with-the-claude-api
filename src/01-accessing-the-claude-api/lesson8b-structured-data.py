"""Building with the Claude API: Lesson 8b - Structured data (structured outputs)

The same task as lesson 8a, getting Claude's answer as JSON that a program can read, done
the current way.

    8a (deprecated):  prefill "```json" + stop sequence  ->  text  ->  json.loads()
    8b (current):     Pydantic model + messages.parse()  ->  response.parsed_output

How structured outputs work:
- A Pydantic model (EventBridgeRule) describes the shape of the answer.
- messages.parse() makes Claude's reply match it: plain JSON, no code fence, no commentary.
- The SDK parses the reply into response.parsed_output, so there is no json.loads() step.

The same pattern is used in lessons 11, 13 and 14.

https://academy.claude.com/courses/building-with-the-claude-api/structured-data

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import anthropic
from pydantic import BaseModel
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


# The shape of the answer: a short EventBridge rule.
class EventPattern(BaseModel):
    source: list[str]
    detail_type: list[str]


class EventBridgeRule(BaseModel):
    name: str
    description: str
    event_pattern: EventPattern
    state: str


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000
user_prompt = "Generate a very short event bridge rule as json"


def main():
    messages = [{"role": "user", "content": user_prompt}]

    log_turn(1, note="structured outputs")
    # parse() sends output_format as output_config with the JSON schema of EventBridgeRule; the log shows
    # Pydantic's version of that schema (the SDK adjusts it slightly before sending).
    log_model_request(
        model=model, max_tokens=max_tokens, messages=messages,
        output_config={"format": {"type": "json_schema", "schema": EventBridgeRule.model_json_schema()}},
    )

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        output_format=EventBridgeRule,
    )

    log_model_response(response)

    # Already parsed and checked against EventBridgeRule: no fence to strip, no json.loads().
    rule = response.parsed_output
    print(f"\n=== ANSWER TO USER ===\n{rule.model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
