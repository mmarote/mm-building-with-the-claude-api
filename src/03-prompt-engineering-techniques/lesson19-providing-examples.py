"""Building with the Claude API: Lesson 19 - Providing examples

We add one example, a sample input with its ideal output, to lesson 18's prompt, so Claude sees
what a good answer looks like, and measure whether the score beats lesson 18.

What changes from lesson 18 (an example added after the guidelines in prompt_template):

    lesson 18:  first line + <athlete_information> + Guidelines
    lesson 19:  first line + <athlete_information> + Guidelines
                + <sample_input> ... </sample_input>
                + <ideal_output> ... </ideal_output>
                + why the output is ideal

Why it helps:
- An example shows the format, length and tone better than rules can.
- One example (one-shot) shows a single pattern; several (multi-shot) cover varied cases.
- Wrap examples in XML tags, say what they are, and explain why the output is good.
- Aim examples at the most common failures. In lesson 17, every answer lost points for being
  long and for totals that did not add up, so the ideal output is one compact table whose numbers
  add up exactly (each meal: kcal = 4 x protein + 4 x carbs + 9 x fat).

About the example:
- The course suggests taking it from a high-scoring eval answer. Our report does not keep the
  answers, so the example was written by hand.
- Its athlete is not one of the test cases. Otherwise Claude would be shown the answer to a case
  it is graded on, and the score would no longer measure the prompt.

The prompt engineering loop:

    set a goal  ->  write a prompt  ->  evaluate  ->  apply a technique  ->  evaluate again
                                           ^                                      |
                                           +--------------------------------------+

One prompt, one technique per lesson:

    Lesson 15  baseline prompt
    Lesson 16  being clear and direct
    Lesson 17  being specific
    Lesson 18  structure with XML tags
    Lesson 19  providing examples         <- this lesson

The code follows the loop, one block per step:

    STEP 1  set a goal       task_description, prompt_inputs_spec, extra_criteria
    STEP 2  write a prompt   prompt_template: the only part lessons 16-19 change
    STEP 3  evaluate         run_prompt, build_dataset (once), then evaluate

- src/helpers/prompt_evaluator.py: the eval pipeline (dataset, grader, report), built from lessons 11-14.
  The same in every lesson.
- "Diff vs previous" in the runner UI shows what changes from one lesson to the next.

The test cases:
- The first run generates them and saves them to
  data/03-prompt-engineering-techniques/meal-plan-dataset.json.
- Later runs and lessons 16-19 reuse that file, so every lesson is graded on the same cases.
- Delete the file to get new ones. The scores of earlier lessons then no longer compare.

https://academy.claude.com/courses/building-with-the-claude-api/providing-examples

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

<athlete_information>
- Height: {height}
- Weight: {weight}
- Goal: {goal}
- Dietary restrictions: {restrictions}
</athlete_information>

Guidelines:
1. Include accurate daily calorie amount
2. Show protein, fat, and carb amounts
3. Specify when to eat each meal
4. Use only foods that fit restrictions
5. List all portion sizes in grams
6. Keep budget-friendly if mentioned

Here is an example input with an ideal response:

<sample_input>
- Height: 172
- Weight: 68
- Goal: Maintain weight and fuel daily training during a competitive cycling season
- Dietary restrictions: No pork or shellfish
</sample_input>

<ideal_output>
Daily total: 2,447 kcal | Protein 142 g | Carbs 301 g | Fat 75 g

| Time  | Meal      | Foods and portions                                                            | kcal | P (g) | C (g) | F (g) |
|-------|-----------|-------------------------------------------------------------------------------|------|-------|-------|-------|
| 07:00 | Breakfast | 80 g rolled oats, 250 ml milk, 120 g banana, 15 g peanut butter               | 642  | 24    | 96    | 18    |
| 10:00 | Snack     | 150 g Greek yogurt, 30 g granola                                              | 248  | 18    | 26    | 8     |
| 13:00 | Lunch     | 150 g chicken breast, 200 g cooked brown rice, 100 g broccoli, 10 g olive oil | 581  | 54    | 53    | 17    |
| 16:30 | Pre-ride  | 70 g whole-grain bread, 20 g honey, 180 g apple                               | 338  | 9     | 71    | 2     |
| 19:30 | Dinner    | 150 g salmon, 250 g boiled potatoes, 150 g mixed salad, 10 g olive oil        | 638  | 37    | 55    | 30    |
</ideal_output>

This output is ideal because it is compact (one table, no extra sections or tips), gives every food
with its portion in grams and its time, respects the restriction, and its numbers add up: each meal's
kcal equals 4 x protein + 4 x carbs + 9 x fat, and the meals add up to the daily total.
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
