"""Building with the Claude API: Lesson 12 - Running the eval

The evaluation pipeline: every test case from the dataset goes through the prompt, and each answer is collected with its test case.

For each test case:

    test case  ->  prompt template  ->  Claude  ->  answer  ->  grade (placeholder: always 10)

- Dataset: data/02-prompt-evaluation/prompt-eval-dataset.json, written by lesson 11.
- Grading: only a placeholder here. The real graders come in lessons 13 and 14.

Known limitations:
- The prompt has no formatting instructions, so Claude's answers are verbose. Later lessons refine it.
- Running the whole dataset takes a while, even with Haiku.

https://academy.claude.com/courses/building-with-the-claude-api/running-the-eval

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import json
from pathlib import Path
import anthropic
from helpers.chat_helper import add_user_message, chat
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000

# The file written by lesson 11, in this section's folder inside data/ at the project root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "02-prompt-evaluation"
dataset_file = DATA_DIR / "prompt-eval-dataset.json"


def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    # ── NOTE: PROMPT UNDER EVALUATION ──────────────────────────────
    # This template is what the eval measures. Later lessons
    # improve this wording and re-run the eval to compare scores.
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""
    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output


def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # TODO - Grading. Placeholder score until the grading lessons.
    score = 10

    return {
        "output": output,
        "test_case": test_case,
        "score": score
    }


def run_eval(dataset):
    """Loads the dataset and runs every test case through the prompt, collecting the results"""
    results = []

    for test_case in dataset:
        log_turn(len(results) + 1, note=test_case["task"])

        result = run_test_case(test_case)
        results.append(result)

    return results


def main():
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = run_eval(dataset)

    print(f"\n=== ANSWER TO USER ===\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
