"""Building with the Claude API: Lesson 61 - Agents and workflows

A workflow is a fixed series of Claude calls that our code runs; here we build the lesson's
evaluator-optimizer workflow: a producer writes, and a grader sends it back until it passes.

Workflow or agent (the course's rule of thumb):
- Workflow: you can picture the exact steps. Our code decides what happens next.
- Agent: the steps are unclear. Claude gets a goal and tools and decides (lesson 65).

The evaluator-optimizer loop:

    task ──► PRODUCER (Claude) ──► draft ──► GRADER (Claude) ──► passed? ──yes──► done
                  ▲                                                │
                  └──────────────── feedback ◄────────── no ───────┘
                  (at most max_rounds rounds, so it always ends)

Two calls per round, each with one job:
- Producer: write the text (on later rounds: fix it, using the grader's feedback).
- Grader: check the text against the criteria, nothing else. Structured outputs (lesson 13)
  make its verdict a Pydantic object: passed (true/false) and a list of problems.

The course vs. this script:
- The course shows this pattern as a diagram with a CAD example (image -> 3D model ->
  rendering -> grade against the image), and no code. A CAD model needs a 3D library and a
  renderer, so ours produces text: a product description with strict criteria.

https://academy.claude.com/courses/building-with-the-claude-api/agents-and-workflows

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from pydantic import BaseModel
import helpers.chat_helper as chat_helper
from helpers.chat_helper import add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_request, log_model_response


model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000
max_rounds = 3  # the loop always ends, even if the grader never passes the text

task = "Write a product description for the TrailCup, a reusable steel camping mug."
criteria = """- Between 40 and 60 words.
- Mentions that it keeps drinks hot for 6 hours.
- Mentions that it weighs 180 grams.
- No exclamation marks.
- Ends with a question to the reader."""


class Verdict(BaseModel):
    """The grader's answer. Structured outputs make Claude's reply match this shape."""
    passed: bool
    problems: list[str]  # empty when passed


def produce(feedback: list[str] | None, draft: str | None) -> str:
    """The producer: writes the text, or fixes the previous draft using the grader's feedback"""
    if draft is None:
        prompt = f"{task}\n\nCriteria:\n{criteria}\n\nAnswer with the description only."
    else:
        problems = "\n".join(f"- {problem}" for problem in feedback)
        prompt = f"""{task}

Criteria:
{criteria}

Your previous draft:
<draft>
{draft}
</draft>

A reviewer found these problems:
{problems}

Rewrite the draft so it meets every criterion. Answer with the description only."""   # <- new
    messages = []
    add_user_message(messages, prompt)
    return text_from_message(send(messages, model=model, max_tokens=max_tokens))


def grade(draft: str) -> Verdict:
    """The grader: checks the draft against the criteria and returns a Verdict"""
    prompt = f"""Check this product description against every criterion. Count words carefully.

<description>
{draft}
</description>

Criteria:
{criteria}

passed is true only if every criterion is met. List each unmet criterion in problems, saying
what is wrong (e.g. "57 words, but ends without a question")."""
    messages = []
    add_user_message(messages, prompt)
    log_model_request(
        model=model, max_tokens=max_tokens, messages=messages,
        output_config={"format": {"type": "json_schema", "schema": Verdict.model_json_schema()}},
    )
    response = chat_helper.client.messages.parse(                       # <- new
        model=model, max_tokens=max_tokens, messages=messages, output_format=Verdict,
    )
    log_model_response(response)
    return response.parsed_output


def main():
    rounds = []
    draft, feedback = None, None
    for round_number in range(1, max_rounds + 1):
        log_banner(f"ROUND {round_number}: PRODUCER")
        draft = produce(feedback, draft)

        log_banner(f"ROUND {round_number}: GRADER")
        verdict = grade(draft)
        rounds.append((draft, verdict))
        if verdict.passed:                                              # <- new
            break
        feedback = verdict.problems                                     # <- new

    print("\n=== THE ROUNDS ===")
    for number, (text, verdict) in enumerate(rounds, start=1):
        result = "passed" if verdict.passed else "failed: " + "; ".join(verdict.problems)
        print(f"  round {number}: {len(text.split())} words, {result}")

    final_draft, final_verdict = rounds[-1]
    if not final_verdict.passed:
        print(f"\n  Not passed after {max_rounds} rounds: the workflow stops and returns the last draft.")
    print(f"\n=== ANSWER TO USER ===\n{final_draft}")


if __name__ == "__main__":
    main()
