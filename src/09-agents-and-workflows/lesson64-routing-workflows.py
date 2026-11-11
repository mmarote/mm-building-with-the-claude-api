"""Building with the Claude API: Lesson 64 - Routing workflows

A router sends each request down one of several specialized paths; here Claude first sorts a
video topic into a genre, and our code then picks that genre's prompt to write the script.

The workflow, for each topic:

    "Python functions" ──► ROUTER ──► "Educational" ──► Educational template ──► script
                           (Claude)     our code looks     (Claude) writes in
                           names the    up the template    that genre's style
                           category

    "surfing"          ──► ROUTER ──► "Entertainment" ──► Entertainment template ──► script

Why route:
- Each genre gets a prompt written and tested for it, instead of one prompt for all.
- The router call is short and cheap; the specialized call does the real work.
- Each route can be its own workflow, with its own prompts or tools.

Our code checks the router's answer: if it is not one of the categories (e.g. a sentence
instead of one word), it falls back to a general template rather than failing.

When to route (from the course): requests vary a lot, the categories are clearly defined,
and Claude can sort them reliably.

The course vs. this script:
- The router prompt is the course's. The course shows only the Educational template, in a
  diagram and no code; the other five templates are ours.
- We add one line to the router prompt (answer with the category only), so our code can
  look up the template.

https://academy.claude.com/courses/building-with-the-claude-api/routing-workflows

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

from helpers.chat_helper import add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner


model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000

topics = ["Python functions", "surfing"]

# One template per route. {topic} is filled in for each request.
STYLE = "Write a script for a 60-second social media video about {topic}. "
templates = {
    "Educational": STYLE + "Make complex information clear and engaging: use relatable examples, "
                           "ask thought-provoking questions, and end with one key takeaway.",
    "Entertainment": STYLE + "Make it fun and high-energy: a hook in the first line, quick scenes, "
                             "and a surprising moment.",
    "Comedy": STYLE + "Make it funny: build to a punchline, use one running gag, keep it light.",
    "Personal vlog": STYLE + "Tell it in the first person, as a day in the creator's life, "
                             "with honest reactions.",
    "Reviews": STYLE + "Review it: what it is, two pros, one con, and a clear verdict.",
    "Storytelling": STYLE + "Tell it as a short story with a beginning, a turning point and an ending.",
}
fallback_template = STYLE + "Make it clear and engaging for a general audience."

# The course's router prompt, plus the last line.
router_prompt = """Categorize the topic of a video into one of the listed categories:
<topic>{topic}</topic>

<categories>
{categories}
</categories>

Answer with the category name only."""


def ask(prompt: str) -> str:
    messages = []
    add_user_message(messages, prompt)
    return text_from_message(send(messages, model=model, max_tokens=max_tokens)).strip()


def route(topic: str) -> str:
    """The router: one short Claude call that names the category"""
    categories = "\n".join(f"- {name}" for name in templates)
    return ask(router_prompt.format(topic=topic, categories=categories))


def main():
    results = []
    for topic in topics:
        log_banner(f"ROUTE: WHICH CATEGORY IS '{topic}'?")
        category = route(topic)                                                   # <- new

        # Our code picks the path. An answer that is not a category gets the general template.
        template = templates.get(category, fallback_template)                    # <- new
        path = category if category in templates else f"fallback ('{category}' is not a category)"

        log_banner(f"WRITE: THE {path.upper()} TEMPLATE FOR '{topic}'")
        script = ask(template.format(topic=topic))
        results.append((topic, path, script))

    print("\n=== THE ROUTES ===")
    for topic, path, _ in results:
        print(f"  {topic:18} -> {path}")

    for topic, path, script in results:
        print("\n=== ANSWER TO USER ===")
        print(f"\n--- {topic} ({path}) ---\n{script}")


if __name__ == "__main__":
    main()
