"""
Shared prompt evaluation pipeline for the prompt engineering lessons (section 3, lessons 15-19).

Lessons 11-14 build an eval step by step: a generated dataset, a prompt run on each test case, and a
model grader. This file packs those steps into one class, so each prompt engineering lesson only
defines its prompt. Everything else stays the same from lesson to lesson, so a change in the score
comes from the prompt alone.

    evaluator = PromptEvaluator(max_concurrent_tasks=5)
    evaluator.generate_dataset(task_description, prompt_inputs_spec, output_file, num_cases=3)
    evaluator.run_evaluation(run_prompt_function, dataset_file, extra_criteria)

Differences from the course's PromptEvaluator:
- Structured outputs instead of assistant prefill, for both the dataset and the grades (see lesson 11).
- The report is printed (the runner UI shows it as the answer), not written to an HTML file.

The log_* calls come from helpers/log_helper.py and only print what is happening.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
import anthropic
from pydantic import BaseModel, Field, create_model
from helpers import log_helper
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_request, log_model_response


# The data folder at the project root, whatever folder the script is started from.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# A test case passes when the grader gives it at least this score (out of 10).
PASS_SCORE = 7


class Grade(BaseModel):
    """The shape of the grader's answer, as in lessons 13 and 14."""
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int


class PromptEvaluator:
    def __init__(self, max_concurrent_tasks=3, model=CHEAPEST):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.model = model
        self.client = anthropic.Anthropic()
        # The dataset and the grades are JSON, which can be long. If a reply is cut off at the
        # limit it can't be parsed. You only pay for the tokens Claude actually writes.
        self.max_tokens = 4000

    # ---- Dataset ---------------------------------------------------------------

    def generate_dataset(self, task_description, prompt_inputs_spec, output_file, num_cases=3):
        """Asks Claude for test cases and saves them as JSON. Returns the list of test cases.

        prompt_inputs_spec maps each prompt input to a description, e.g. {"height": "Athlete's height in cm"}.
        Each test case gets a value for every input, plus the criteria a good answer must meet.
        """
        # A Pydantic model built from the spec, so Claude must return exactly these inputs.
        PromptInputs = create_model(
            "PromptInputs", **{name: (str, Field(description=text)) for name, text in prompt_inputs_spec.items()}
        )
        TestCase = create_model(
            "TestCase",
            scenario=(str, Field(description="A short description of this test case")),
            prompt_inputs=(PromptInputs, ...),
            solution_criteria=(list[str], Field(description="1-4 things a good answer must do")),
        )
        Dataset = create_model("Dataset", cases=(list[TestCase], ...))

        inputs_text = "\n".join(f"- {name}: {text}" for name, text in prompt_inputs_spec.items())
        prompt = f"""Generate {num_cases} test cases for evaluating a prompt that does this task:

<task_description>
{task_description}
</task_description>

The prompt takes these inputs:
<prompt_inputs>
{inputs_text}
</prompt_inputs>

Make the test cases varied and realistic, including some harder ones. For each test case, give a value
for every input and the criteria a good answer must meet."""

        messages = [{"role": "user", "content": prompt}]

        log_banner("GENERATE THE DATASET")
        # parse() sends output_format as output_config with the JSON schema of Dataset; the log shows
        # Pydantic's version of that schema (the SDK adjusts it slightly before sending).
        log_model_request(
            model=self.model, max_tokens=self.max_tokens, messages=messages,
            output_config={"format": {"type": "json_schema", "schema": Dataset.model_json_schema()}},
        )

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            output_format=Dataset,
        )

        log_model_response(response)

        # The task description is saved with each test case, because the grader needs it.
        dataset = [{"task_description": task_description, **case.model_dump()} for case in response.parsed_output.cases]

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
        print(f"\nSaved {len(dataset)} test cases to {output_file}")
        return dataset

    # ---- Grading ---------------------------------------------------------------

    def grade(self, test_case, output, extra_criteria=None):
        """Asks Claude to grade one output, returns a Grade (strengths, weaknesses, reasoning, score)."""
        criteria = "\n".join(f"- {item}" for item in test_case["solution_criteria"])
        extra = f"\n\nThe answer must also meet these criteria:\n<extra_criteria>\n{extra_criteria}\n</extra_criteria>" if extra_criteria else ""
        eval_prompt = f"""You are an expert evaluator. Grade this AI-generated answer.

<task_description>
{test_case["task_description"]}
</task_description>

<prompt_inputs>
{json.dumps(test_case["prompt_inputs"], indent=2)}
</prompt_inputs>

<solution_criteria>
{criteria}
</solution_criteria>{extra}

<answer>
{output}
</answer>

Provide your evaluation with:
- strengths: 1-3 key strengths
- weaknesses: 1-3 key areas for improvement
- reasoning: a concise explanation of your assessment
- score: a number between 1 and 10. Any missed criterion lowers the score a lot."""

        messages = [{"role": "user", "content": eval_prompt}]
        # parse() sends output_format as output_config with the JSON schema of Grade; the log shows
        # Pydantic's version of that schema (the SDK adjusts it slightly before sending).
        log_model_request(
            model=self.model, max_tokens=self.max_tokens, messages=messages,
            output_config={"format": {"type": "json_schema", "schema": Grade.model_json_schema()}},
        )

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            output_format=Grade,
        )

        log_model_response(response)
        return response.parsed_output

    # ---- Evaluation ------------------------------------------------------------

    def run_evaluation(self, run_prompt_function, dataset_file, extra_criteria=None):
        """Runs the prompt on every test case, grades each output, and prints a report. Returns the results.

        run_prompt_function(prompt_inputs) gets one test case's inputs and returns Claude's answer.
        """
        dataset_file = Path(dataset_file)
        if not dataset_file.exists():
            raise FileNotFoundError(f"{dataset_file} not found. Run lesson 15 first, it generates the dataset.")
        dataset = json.loads(dataset_file.read_text(encoding="utf-8"))

        def run_test_case(test_case):
            output = run_prompt_function(test_case["prompt_inputs"])
            grade = self.grade(test_case, output, extra_criteria)
            print(f"   graded: {test_case['scenario']} -> {grade.score}/10", flush=True)
            return {"test_case": test_case, "output": output, **grade.model_dump()}

        # The test cases run in parallel, so their request logs would mix together. The logs are
        # switched off for this part; the report at the end shows each case.
        log_banner(f"RUN AND GRADE {len(dataset)} TEST CASES, {self.max_concurrent_tasks} AT A TIME (logs off)")
        logs_were_on = log_helper.ENABLED
        log_helper.ENABLED = False
        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
                results = list(executor.map(run_test_case, dataset))
        finally:
            log_helper.ENABLED = logs_were_on

        self.print_report(results)
        return results

    def print_report(self, results):
        """Prints each test case's score and reasoning, then the average score and the pass rate."""
        lines = []
        for number, result in enumerate(results, start=1):
            lines.append(f"{number}. {result['test_case']['scenario']}")
            lines.append(f"   score:      {result['score']}/10")
            lines.append(f"   reasoning:  {result['reasoning']}")
            lines.append(f"   weaknesses: {'; '.join(result['weaknesses'])}")
            lines.append("")

        average_score = mean(result["score"] for result in results)
        passed = sum(1 for result in results if result["score"] >= PASS_SCORE)
        lines.append(f"Average score: {average_score:.2f} / 10")
        lines.append(f"Pass rate:     {passed} of {len(results)} ({passed / len(results):.0%}, score {PASS_SCORE} or more)")

        report = "\n".join(lines)
        print(f"\n=== ANSWER TO USER ===\n{report}")
