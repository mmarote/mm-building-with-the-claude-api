"""Building with the Claude API: Lesson 16 - Being clear and direct

We rewrite the first line of lesson 15's prompt as a clear, direct instruction
and measure whether the score beats lesson 15's baseline.

What changes from lesson 15 (only the first line of prompt_template):

    lesson 15:  What should this person eat?
    lesson 16:  Generate a one-day meal plan for an athlete that meets their dietary restrictions.

Why it helps:
- The first line of a prompt has a large effect on the result.
- Clear: simple language, the task stated without hedging.
- Direct: an instruction, not a question, starting with a verb (Write, Create, Generate).
- The task and its constraints come first: a one-day plan, for an athlete, meeting the restrictions.

The prompt engineering loop:

    set a goal  ->  write a prompt  ->  evaluate  ->  apply a technique  ->  evaluate again
                                           ^                                      |
                                           +--------------------------------------+

One prompt, one technique per lesson:

    Lesson 15  baseline prompt
    Lesson 16  being clear and direct     <- this lesson
    Lesson 17  being specific
    Lesson 18  structure with XML tags
    Lesson 19  providing examples

The code follows the loop, one block per step:

    STEP 1  set a goal       task_description, prompt_inputs_spec, extra_criteria
    STEP 2  write a prompt   prompt_template: the only part lessons 16-19 change
    STEP 3  evaluate         run_prompt, build_dataset (once), then evaluate

- src/helpers/prompt_evaluator.py: the eval pipeline (dataset, grader, report), built from
  lessons 11-14. The same in every lesson.
- "Diff vs previous" in the runner UI shows what changes from one lesson to the next.

The test cases:
- The first run generates them and saves them to
  data/03-prompt-engineering-techniques/meal-plan-dataset.json.
- Later runs and lessons 16-19 reuse that file, so every lesson is graded on the same cases.
- Delete the file to get new ones. The scores of earlier lessons then no longer compare.

https://academy.claude.com/courses/building-with-the-claude-api/being-clear-and-direct

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from helpers.chat_helper import add_user_message, chat
from helpers.models import CHEAPEST
from helpers.prompt_evaluator import PromptEvaluator, DATA_DIR


model = CHEAPEST
# Claude thinks before it answers, and thinking counts toward max_tokens. Too low a limit cuts the
# meal plan off (a lower score for the wrong reason) or leaves no answer at all.
# You only pay for the tokens Claude actually uses.
max_tokens = 16000

dataset_file = DATA_DIR / "03-prompt-engineering-techniques" / "meal-plan-dataset.json"


# ══ STEP 1: SET A GOAL ══════════════════════════════════════════════════════
# What a good answer looks like. The same in lessons 15-19, so the scores compare.
# These go to the evaluator only (dataset and grader), not to Claude when it answers.

# The task the prompt must do. Used to generate the test cases and to grade the answers.
task_description = "Write a compact, concise 1 day meal plan for a single athlete"

# The inputs the prompt receives. Each test case gets a value for every one of them.
prompt_inputs_spec = {
    "height": "Athlete's height in cm",
    "weight": "Athlete's weight in kg",
    "goal": "Goal of the athlete",
    "restrictions": "Dietary restrictions of the athlete",
}

# What every answer must include. The grader checks these on top of each test case's own criteria.
extra_criteria = """
The output should include:
- Daily caloric total
- Macronutrient breakdown
- Meals with exact foods, portions, and timing
"""


# ══ STEP 2: WRITE A PROMPT ══════════════════════════════════════════════════
# ── PROMPT UNDER EVALUATION: the only thing that changes in lessons 16-19 ──
# {height}, {weight}, {goal} and {restrictions} are filled in from each test case.
# Literal curly braces (e.g. in a JSON example) must be written doubled: {{ and }}.
prompt_template = """
Generate a one-day meal plan for an athlete that meets their dietary restrictions.

- Height: {height}
- Weight: {weight}
- Goal: {goal}
- Dietary restrictions: {restrictions}
"""


# ══ STEP 3: EVALUATE ════════════════════════════════════════════════════════

def run_prompt(prompt_inputs):
    """Fills the template (step 2) with one test case's inputs, then returns Claude's answer"""
    prompt = prompt_template.format(**prompt_inputs)
    messages = []
    add_user_message(messages, prompt)
    return chat(messages, model=model, max_tokens=max_tokens)


def build_dataset(evaluator):
    """Turns the goal (step 1) into test cases. Runs once; skipped when the dataset file exists."""
    if dataset_file.exists():
        print(f"Using the saved test cases in {dataset_file} (delete the file to generate new ones).")
        return

    evaluator.generate_dataset(
        task_description=task_description,
        prompt_inputs_spec=prompt_inputs_spec,
        output_file=dataset_file,
        num_cases=3,
    )


def evaluate(evaluator):
    """Runs the prompt (step 2) on every test case, grades each answer, and prints the report."""
    evaluator.run_evaluation(
        run_prompt_function=run_prompt,
        dataset_file=dataset_file,
        extra_criteria=extra_criteria,
    )


def main():
    evaluator = PromptEvaluator(max_concurrent_tasks=3)
    build_dataset(evaluator)
    evaluate(evaluator)


if __name__ == "__main__":
    main()
