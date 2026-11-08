"""Building with the Claude API: Lesson 11 - Generating test datasets

Claude writes the test cases for our eval, because writing dozens of them by hand is slow.

    Claude (Haiku)
      ->  list of test cases
      ->  saved to data/02-prompt-evaluation/prompt-eval-dataset.json
      ->  read by lessons 12-14

What an evaluation dataset is:
- A list of test inputs. We run each one through our prompt and check the results.
- Haiku is a good choice to write it: the data is only used for testing.
- Each test case has a "task" and a "format" (python, json or regex). Lesson 14 uses the format.

How the answer comes back:
- A Pydantic model (TestDataset) describes the shape of the answer.
- messages.parse() checks Claude's reply against it.

Note: the course uses assistant prefill and a stop sequence. Prefill is deprecated and not supported
on current models, so we use structured outputs instead, the current recommended method. The result
is the same list of tasks.

https://academy.claude.com/courses/building-with-the-claude-api/generating-test-datasets

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import json
from pathlib import Path
from typing import Literal
import anthropic
from pydantic import BaseModel
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_request, log_model_response


class TestCase(BaseModel):
    task: str
    # The kind of code the task asks for. Lesson 14 uses it to pick the syntax validator.
    format: Literal["python", "json", "regex"]


class TestDataset(BaseModel):
    tasks: list[TestCase]


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000
# This section's folder inside data/ at the project root, whatever folder the script is started from.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "02-prompt-evaluation"
dataset_file = DATA_DIR / "prompt-eval-dataset.json"


def generate_dataset(count=3):
    prompt = f"""Generate {count} evaluation tasks for a prompt that helps users write AWS output in one of
three formats: Python code, JSON configuration, or regular expressions.

Each task must need a single Python function, a single JSON object, or a single regex, and needs little code.
Each task has two fields: "task", with the description of the task, and "format", the kind of code
the task needs: "python", "json" or "regex"."""

    messages = [{"role": "user", "content": prompt}]

    log_banner("GENERATE TEST CASES")
    # parse() sends output_format as output_config with the JSON schema of TestDataset; the log shows
    # Pydantic's version of that schema (the SDK adjusts it slightly before sending).
    log_model_request(
        model=model, max_tokens=max_tokens, messages=messages,
        output_config={"format": {"type": "json_schema", "schema": TestDataset.model_json_schema()}},
    )

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        output_format=TestDataset,
    )

    log_model_response(response)
    return response.parsed_output


def main():
    dataset = generate_dataset()
    tasks = [case.model_dump() for case in dataset.tasks]
    print(f"\n=== ANSWER TO USER ===\n{json.dumps(tasks, indent=2)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    print(f"\nSaved {len(tasks)} test cases to {dataset_file}")


if __name__ == "__main__":
    main()
