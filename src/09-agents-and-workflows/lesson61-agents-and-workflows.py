"""Building with the Claude API: Lesson 61 - Agents and workflows

The course explains workflows with a diagram and no code; this script is our own
illustration of its evaluator-optimizer example: a producer draws an ASCII rocket, and a
grader sends it back until it meets every rule.

Workflow or agent (the course's rule of thumb):
- Workflow: you can picture the exact steps. Our code decides what happens next.
- Agent: the steps are unclear. Claude gets a goal and tools and decides (lesson 65).

The evaluator-optimizer loop:

    task ──► PRODUCER (Claude) ──► draft ──► GRADER (Claude) ──► passed? ──yes──► done
                  ▲                                                │
                  └──────────────── feedback ◄────────── no ───────┘
                  safety limit: at most max_rounds rounds (3), then it stops
                  and returns the last draft, so it never loops forever

Two calls per round, each with one job:
- Producer: draw the rocket (on later rounds: fix it, using the grader's feedback).
- Grader: check the drawing against the rules, nothing else. Structured outputs (lesson 13)
  make its verdict a Pydantic object: passed (true/false) and a list of problems.

The task: a rocket with strict, checkable rules. One drawing that meets them all:

           ^
          / \
         /   \
        |     |
        |  O  |          12 lines, at most 15 characters wide,
        |     |          symmetric, a nose cone, one window (O),
        |     |          two fins, and flames made of *
       /|     |\
      / |_____| \
         *   *
        * * * *
         *   *

Demo device: in round 1 the producer gets only "draw a rocket", without the rules, so it
almost always fails and you can watch the loop fix it. Only the grader knows the rules at
first, like a reviewer who holds the spec. From round 2 on, the producer gets the rules too.

The course vs. this script:
- The course shows this pattern as a diagram with a CAD example (image -> 3D model ->
  rendering -> grade against the image), and no code. A CAD model needs a 3D library and a
  renderer, so ours draws ASCII art instead.
- Leaving the rules out of round 1 is our addition, so the run shows more than one round.

https://academy.claude.com/courses/building-with-the-claude-api/agents-and-workflows

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from pydantic import BaseModel
import helpers.chat_helper as chat_helper
from helpers.chat_helper import add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_request, log_model_response


# Not in the course: the course shows this workflow only as a diagram. Everything below is
# our illustration of it, with an ASCII rocket in place of the course's CAD model.

model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000
# Safety limit: the loop ends after max_rounds, even if the grader never passes the drawing.
# Worst case: max_rounds x 2 Claude calls (producer + grader).
max_rounds = 3                                                          # <- new

task = "Draw a rocket in ASCII art."
rules = """- Exactly 12 lines.
- No line wider than 15 characters.
- Left-right symmetric around a vertical center line (/ mirrors \\, ( mirrors )).
- A pointed nose cone at the top.
- Exactly one window, drawn as the letter O.
- Two fins at the bottom of the body, drawn with / and \\.
- Flames below the rocket: at least two lines made only of * and spaces.
- The drawing only: no title, no explanation, no code fences."""


class Verdict(BaseModel):
    """The grader's answer. Structured outputs make Claude's reply match this shape."""
    passed: bool
    problems: list[str]  # empty when passed


def produce(feedback: list[str] | None, draft: str | None) -> str:
    """The producer: draws the rocket, or fixes the last drawing using the grader's feedback"""
    if draft is None:
        # Demo device: round 1 does not get the rules, so it almost always fails.
        prompt = f"{task}\n\nAnswer with the drawing only."                # <- new
    else:
        problems = "\n".join(f"- {problem}" for problem in feedback)
        prompt = f"""{task}

Rules:
{rules}

Your previous drawing:
<drawing>
{draft}
</drawing>

A reviewer found these problems:
{problems}

Redraw it so it meets every rule. Answer with the drawing only."""          # <- new
    messages = []
    add_user_message(messages, prompt)
    return text_from_message(send(messages, model=model, max_tokens=max_tokens))


def grade(draft: str) -> Verdict:
    """The grader: checks the drawing against the rules and returns a Verdict"""
    prompt = f"""Check this ASCII drawing against every rule.
Count lines and characters carefully.

<drawing>
{draft}
</drawing>

Rules:
{rules}

passed is true only if every rule is met. List each broken rule in problems, saying what
is wrong (e.g. "14 lines, but the rule is exactly 12")."""
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


def size(drawing: str) -> str:
    """Lines and width of a drawing, measured by our code (compare with the grader)"""
    lines = drawing.splitlines()
    width = max((len(line) for line in lines), default=0)
    return f"{len(lines)} lines, up to {width} wide"


def main():
    log_banner("NOT IN THE COURSE: OUR ILLUSTRATION OF THE LESSON'S DIAGRAM")
    print(f"Evaluator-optimizer loop, at most {max_rounds} rounds (safety limit).")

    rounds = []
    draft, feedback = None, None
    for round_number in range(1, max_rounds + 1):                       # <- new
        log_banner(f"ROUND {round_number}: PRODUCER")
        draft = produce(feedback, draft)

        log_banner(f"ROUND {round_number}: GRADER")
        verdict = grade(draft)
        rounds.append((draft, verdict))
        if verdict.passed:                                              # <- new
            break
        feedback = verdict.problems                                     # <- new

    print("\n=== THE ROUNDS ===")
    for number, (drawing, verdict) in enumerate(rounds, start=1):
        result = "PASSED" if verdict.passed else "FAILED"
        print(f"\n--- round {number}: {size(drawing)}: {result} ---")
        print("\n".join("    " + line for line in drawing.splitlines()))
        for problem in verdict.problems:
            print(f"  - {problem}")

    # The answer is the last round's drawing, printed above.
    if not rounds[-1][1].passed:
        print(f"\n  Not passed after {max_rounds} rounds: "
              "the workflow stops and returns the last draft.")


if __name__ == "__main__":
    main()
