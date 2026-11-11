"""Building with the Claude API: Lesson 63 - Chaining workflows

A chain splits a task into steps that run one after another, each with one job; here a draft
step writes an article, our code checks it, and a revision step fixes only what went wrong.

The chain (the course's example):

    STEP 1  Claude    write the article: topic + every constraint in one long prompt
               │      (with many rules at once, some tend to get missed)
               ▼
    CHECK   our code  look for what the rules forbid: emojis, mentions of AI, cliches
               │      (no Claude call: plain Python between two steps)
               ▼
    STEP 2  Claude    revise the draft: a short prompt with only the fixes
               ▼
    CHECK   our code  the same check on the final text

Why it works: step 2 has one job (fix these problems), so Claude does not have to balance
writing good content against every rule at once.

When to chain (from the course):
- A task with many requirements, where Claude keeps missing some of them.
- Output that needs checking or processing between steps.
- Steps that are clearer, and easier to test, on their own.

The course vs. this script:
- The course shows the two prompts in a diagram and no code. Its revision prompt starts with
  "Revise the article provided below." and lists three fixes; ours follows it.
- The check between the steps is ours. The course mentions that a chain can process the
  output between steps; a real app could also skip step 2 when the check finds nothing.

https://academy.claude.com/courses/building-with-the-claude-api/chaining-workflows

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import re
from helpers.chat_helper import add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner


model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000

# STEP 1: the long prompt, with every constraint at once.
draft_prompt = """Write a short technical article (about 300 words) about Python list comprehensions,
for developers who know Python basics.

Rules:
- Do not mention that the article was written by an AI or an assistant.
- Do not use any emojis.
- Avoid cliches and casual language such as "dive into", "game-changer" or "let's explore".
- Use the professional tone of an experienced technical writer.
- Include one short code example."""

# STEP 2: a short prompt with one job: fix the draft (after the course's revision prompt).
revision_prompt = """Revise the article provided below. Make these fixes:
1. Remove any reference to the article being written by an AI or an assistant.
2. Remove all emojis.
3. Replace cringey or cliched writing with the phrasing of an experienced technical writer.

Keep everything else, including the code example. Answer with the revised article only.

<article>
{article}
</article>"""

# What the check looks for. It is plain code, so it costs nothing and gives the same answer every time.
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
AI_MENTIONS = ["as an ai", "ai assistant", "language model", "i'm an ai", "i am an ai"]
CLICHES = ["dive into", "deep dive", "game-changer", "game changer", "let's explore", "unlock the power",
           "in today's fast-paced", "buckle up", "supercharge"]


def ask(prompt: str) -> str:
    messages = []
    add_user_message(messages, prompt)
    return text_from_message(send(messages, model=model, max_tokens=max_tokens))


def check(article: str) -> list[str]:
    """The step between the Claude calls: the rule breaks plain code can find"""      # <- new
    lower = article.lower()
    problems = [f"emoji: {emoji}" for emoji in sorted(set(EMOJI.findall(article)))]
    problems += [f"mentions AI: '{phrase}'" for phrase in AI_MENTIONS if phrase in lower]
    problems += [f"cliche: '{phrase}'" for phrase in CLICHES if phrase in lower]
    return problems


def show_check(title: str, problems: list[str]):
    print(f"\n{title}: {len(problems)} problem(s) found")
    for problem in problems:
        print(f"  - {problem}")


def main():
    log_banner("STEP 1: WRITE THE DRAFT (one long prompt, every rule at once)")
    draft = ask(draft_prompt)

    log_banner("CHECK: OUR CODE LOOKS FOR RULE BREAKS IN THE DRAFT")
    draft_problems = check(draft)                                             # <- new
    show_check("Draft", draft_problems)

    log_banner("STEP 2: REVISE THE DRAFT (a short prompt with only the fixes)")
    final = ask(revision_prompt.format(article=draft))                        # <- new

    log_banner("CHECK: THE SAME CHECK ON THE FINAL TEXT")
    final_problems = check(final)
    show_check("Final", final_problems)

    print("\n=== THE CHAIN ===")
    print(f"  draft:  {len(draft.split())} words, {len(draft_problems)} problem(s) found by the check")
    print(f"  final:  {len(final.split())} words, {len(final_problems)} problem(s) found by the check")
    print("  (The check only finds what code can find: e.g. not the tone. Step 2 also works on that.)")

    print(f"\n=== ANSWER TO USER ===\n{final}")


if __name__ == "__main__":
    main()
