"""Building with the Claude API: Lesson 13 - Model based grading

Lesson 12 gave every answer a placeholder score of 10; here a second call to Claude grades each answer.

For each test case:

    task  ->  Claude  ->  answer  ->  Claude as grader  ->  strengths, weaknesses, reasoning, score (1-10)

Then the average score is printed: the baseline to beat when we improve the prompt.

Why the grader writes strengths, weaknesses and reasoning before the score:
- It makes the grader think it through, so the scores don't all land around 6.

The three kinds of graders:
- Code graders:  length, required words, valid syntax     (lesson 14)
- Model graders: instruction following, completeness,     (this lesson)
                 helpfulness
- Human graders: flexible, but slow

Notes:
- Dataset: data/02-prompt-evaluation/prompt-eval-dataset.json, written by lesson 11.
- The course gets the grader's JSON with assistant prefill and a stop sequence. Prefill is deprecated
  and not supported on current models, so, as in lesson 11, we use structured outputs instead: a
  Pydantic model (Grade) describes the answer, and messages.parse() checks Claude's reply against it.

https://academy.claude.com/courses/building-with-the-claude-api/model-based-grading

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import json
from pathlib import Path
from statistics import mean
import anthropic
from pydantic import BaseModel
from helpers.chat_helper import add_user_message, chat
from helpers.models import CHEAPEST
from helpers.log_helper import log_turn, log_model_request, log_model_response


class Grade(BaseModel):
    """The shape of the grader's answer. Pydantic checks the types, so score is always an int."""
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int


client = anthropic.Anthropic()

model = CHEAPEST
max_tokens = 1000
# The grader's JSON can be longer than the answer it grades; if it is cut off at the limit,
# it can't be parsed. You only pay for the tokens Claude actually writes.
grader_max_tokens = 4000

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
    output = chat(messages, model=model, max_tokens=max_tokens)
    return output


def grade_by_model(test_case, output):
    """Asks Claude to grade the output, returns a Grade (strengths, weaknesses, reasoning, score)"""
    # The tags keep the task and the solution apart, even when the solution is a long code block.
    eval_prompt = f"""
    You are an expert code reviewer. Evaluate this AI-generated solution.

    <task>{test_case["task"]}</task>
    <solution>{output}</solution>

    Provide your evaluation as a structured JSON object with:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement  
    - "reasoning": A concise explanation of your assessment
    - "score": A number between 1-10
"""
    messages = []
    add_user_message(messages, eval_prompt)

    # parse() sends output_format as output_config with the JSON schema of Grade; the log shows
    # Pydantic's version of that schema (the SDK adjusts it slightly before sending).
    log_model_request(
        model=model, max_tokens=grader_max_tokens, messages=messages,
        output_config={"format": {"type": "json_schema", "schema": Grade.model_json_schema()}},
    )

    # Structured outputs instead of the course's prefill: Claude's reply must match the Grade model.
    response = client.messages.parse(
        model=model,
        max_tokens=grader_max_tokens,
        messages=messages,
        output_format=Grade,
    )

    log_model_response(response)
    return response.parsed_output


def run_test_case(test_case):
    """Runs the prompt on one test case, then grades the output with the model grader"""
    output = run_prompt(test_case)

    # Grade the output
    model_grade = grade_by_model(test_case, output)
    score = model_grade.score
    reasoning = model_grade.reasoning
    strengths = model_grade.strengths
    weaknesses = model_grade.weaknesses

    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


def run_eval(dataset):
    """Runs and grades every test case, then prints the average score across all of them"""
    results = []

    for test_case in dataset:
        log_turn(len(results) + 1, note=test_case["task"])

        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(f"\nAverage score: {average_score:.1f}")

    return results


def main():
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = run_eval(dataset)

    print(f"\n=== EVAL REPORT ===\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
