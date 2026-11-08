"""Building with the Claude API: Lesson 15 - Prompt engineering

We write a first, deliberately simple prompt for a one-day athlete meal plan
and measure its score: the baseline that lessons 16-19 try to beat.

The prompt engineering loop:

    set a goal  ->  write a prompt  ->  evaluate  ->  apply a technique  ->  evaluate again
                                           ^                                      |
                                           +--------------------------------------+

One prompt, one technique per lesson:

    Lesson 15  baseline prompt            <- this lesson
    Lesson 16  being clear and direct
    Lesson 17  being specific
    Lesson 18  structure with XML tags
    Lesson 19  providing examples

How the code is split:
- src/helpers/prompt_evaluator.py: the eval pipeline (dataset, grader, report), built from lessons 11-14.
  The same in every lesson.
- This file: the prompt in run_prompt. The only part that changes from lesson to lesson.
  "Diff vs previous" in the runner UI shows the change.

The test cases:
- The first run generates them and saves them to
  data/03-prompt-engineering-techniques/meal-plan-dataset.json.
- Later runs and lessons 16-19 reuse that file, so every lesson is graded on the same cases.
- Delete the file to get new ones. The scores of earlier lessons then no longer compare.

https://academy.claude.com/courses/building-with-the-claude-api/prompt-engineering

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from helpers.chat_helper import add_user_message, chat
from helpers.models import CHEAPEST
from helpers.prompt_evaluator import PromptEvaluator, DATA_DIR


model = CHEAPEST
# A meal plan can be long; a cut-off answer would get a lower score for the wrong reason.
max_tokens = 2000

dataset_file = DATA_DIR / "03-prompt-engineering-techniques" / "meal-plan-dataset.json"

# The same in every lesson 15-19, so the scores compare.
task_description = "Write a compact, concise 1 day meal plan for a single athlete"
prompt_inputs_spec = {
    "height": "Athlete's height in cm",
    "weight": "Athlete's weight in kg",
    "goal": "Goal of the athlete",
    "restrictions": "Dietary restrictions of the athlete",
}
extra_criteria = """
The output should include:
- Daily caloric total
- Macronutrient breakdown
- Meals with exact foods, portions, and timing
"""


def run_prompt(prompt_inputs):
    """Merges the prompt and the test case inputs, then returns Claude's answer"""
    # ── NOTE: PROMPT UNDER EVALUATION ──────────────────────────────
    # This is the only part that changes in lessons 16-19.
    prompt = f"""
What should this person eat?

- Height: {prompt_inputs["height"]}
- Weight: {prompt_inputs["weight"]}
- Goal: {prompt_inputs["goal"]}
- Dietary restrictions: {prompt_inputs["restrictions"]}
"""
    messages = []
    add_user_message(messages, prompt)
    return chat(messages, model=model, max_tokens=max_tokens)


def main():
    evaluator = PromptEvaluator(max_concurrent_tasks=5)

    # Only the first run pays for the dataset; after that the saved file is reused.
    if not dataset_file.exists():
        evaluator.generate_dataset(
            task_description=task_description,
            prompt_inputs_spec=prompt_inputs_spec,
            output_file=dataset_file,
            num_cases=3,
        )

    evaluator.run_evaluation(
        run_prompt_function=run_prompt,
        dataset_file=dataset_file,
        extra_criteria=extra_criteria,
    )


if __name__ == "__main__":
    main()
