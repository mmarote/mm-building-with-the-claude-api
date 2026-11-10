"""Building with the Claude API: Lesson 40 - Image support

Claude can read images: we send the image as a block next to the question, and we try a plain question and a step-by-step prompt to count marbles.

The user message (one message, two blocks):

    [ image block   {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": <the file, base64>}}
      text block    {"type": "text", "text": "How many marbles ...?"} ]

    Claude replies with text, as usual.

Limits (see the docs for the exact numbers):
- A maximum number of images per request, and a maximum size per image (after base64).
- Smaller maximum dimensions when you send many images at once.
- An image costs tokens by its size (larger image = more tokens).
- Instead of base64 you can send a URL: {"type": "url", "url": "https://..."}.

Prompting with images: the same techniques as with text (lessons 15-19).
- A bare question ("how many?") can give a wrong answer.
- Give a method, step by step; check the result another way; define the output.
- Also useful: an example image with a known answer, and a fixed scale for ratings.

The image: data/06-features-of-claude/marbles.png, 23 marbles drawn by us (the course does not
provide its image), so we know the right answer.

https://academy.claude.com/courses/building-with-the-claude-api/image-support

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import base64
from pathlib import Path
from helpers.chat_helper import add_user_message, chat
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner


image_file = Path(__file__).resolve().parents[2] / "data" / "06-features-of-claude" / "marbles.png"
actual_count = 23  # how many marbles we drew

model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000

# A bare question.
simple_prompt = "How many marbles are in this image?"

# The course's step-by-step prompt: a method, then a check by a different method.
step_by_step_prompt = """Analyze this image of marbles and determine the exact count using this methodology:
1. Begin by identifying each unique marble one at a time. Assign each a number as you identify it.
2. Verify your result by counting with a different method. Start from the bottom-left corner and work row by row, from left to right.

What is the exact, verified number of marbles in this image?"""


def ask_about_image(image_bytes, prompt):
    """Sends the image and the prompt in one user message; returns Claude's answer"""
    messages = []
    add_user_message(messages, [
        # Image block
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_bytes,
            },
        },
        # Text block
        {
            "type": "text",
            "text": prompt,
        },
    ])
    return chat(messages, model=model, max_tokens=max_tokens)


def main():
    # Read the file and encode it as base64 text.
    with open(image_file, "rb") as f:
        image_bytes = base64.standard_b64encode(f.read()).decode("utf-8")

    log_banner("1. A BARE QUESTION")
    simple_answer = ask_about_image(image_bytes, simple_prompt)

    log_banner("2. A STEP-BY-STEP PROMPT")
    step_by_step_answer = ask_about_image(image_bytes, step_by_step_prompt)

    print(f"\n=== ANSWER TO USER ===\nThe image has {actual_count} marbles.")
    print(f"\n--- bare question ---\n{simple_answer}")
    print(f"\n--- step-by-step prompt ---\n{step_by_step_answer}")


if __name__ == "__main__":
    main()
