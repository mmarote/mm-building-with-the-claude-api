"""Building with the Claude API: Lesson 14 - Code based grading

Lesson 13 graded each answer with a model grader only; this lesson adds a code grader that checks the syntax.

For each test case:

    answer  ->  model grader (lesson 13)             ->  model score   (1-10)  \
            ->  code grader: does it parse as its   ->  syntax score  (0 or 10) >  average = final score
                "format" (python, json or regex)?                              /

The code grader picks its validator from the test case's "format" field:
- "python" -> ast.parse()
- "json"   -> json.loads()
- "regex"  -> re.compile()

What is new compared to lesson 13 (marked "NEW IN LESSON 14" in the code):
- the imports ast and re
- validate_json, validate_python, validate_regex and grade_syntax
- run_test_case combines the model score and the syntax score
- the prompt in run_prompt asks for bare code, with no explanation and no Markdown code fences

https://academy.claude.com/courses/building-with-the-claude-api/code-based-grading

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

# ── NEW IN LESSON 14: ast parses Python code, re compiles regular expressions ──
import ast
import re
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
    # ── NEW IN LESSON 14: ask for bare code, so the syntax validators can parse it.
    # The course also prefills "```code" with a stop sequence; prefill doesn't work on
    # current models, so we ask for no Markdown code fences instead. ──
    prompt = f"""
Please solve the following task:

{test_case["task"]}

* Respond only with Python, JSON, or a plain Regex
* Do not add any comments or commentary or explanation
* Do not wrap the answer in Markdown code fences
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

    log_model_request(model=model, max_tokens=grader_max_tokens, messages=messages)

    # Structured outputs instead of the course's prefill: Claude's reply must match the Grade model.
    response = client.messages.parse(
        model=model,
        max_tokens=grader_max_tokens,
        messages=messages,
        output_format=Grade,
    )

    log_model_response(response)
    return response.parsed_output


# ── NEW IN LESSON 14: code grader ─────────────────────────────────────────
# Each validator tries to parse the output: 10 if it parses, 0 if it doesn't.

def validate_json(text):
    """10 if the text is valid JSON, otherwise 0"""
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    """10 if the text is valid Python syntax, otherwise 0"""
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    """10 if the text is a valid regular expression, otherwise 0"""
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(output, test_case):
    """Picks the validator from the test case's format field and returns its score"""
    code_format = test_case["format"]
    if code_format == "json":
        return validate_json(output)
    elif code_format == "python":
        return validate_python(output)
    else:
        return validate_regex(output)

# ── END NEW IN LESSON 14 ──────────────────────────────────────────────────


def run_test_case(test_case):
    """Runs the prompt on one test case, then grades the output with the model grader and the code grader"""
    output = run_prompt(test_case)

    # Grade the output
    model_grade = grade_by_model(test_case, output)
    # ── NEW IN LESSON 14: combine the model score and the syntax score ──
    model_score = model_grade.score
    syntax_score = grade_syntax(output, test_case)
    score = (model_score + syntax_score) / 2
    # ── END NEW IN LESSON 14 ──
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

    print(f"\n=== ANSWER TO USER ===\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
