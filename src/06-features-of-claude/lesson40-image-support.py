"""Building with the Claude API: Lesson 40 - Image support

Claude can read images: we send the image as a block next to the question, and we try
a plain question and a step-by-step prompt to count marbles.

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

Beyond the course: what the image costs (step 3 and the TOKENS table; not in the course lesson).
- About width x height / 750 tokens. Our 900x600 image: ~720 tokens.
- The usage counts the image inside input_tokens, without a separate number; this script finds
  the image's share with client.messages.count_tokens (free, no answer generated).

The image: data/06-features-of-claude/marbles.png, 23 marbles drawn by us (the course does not
provide its image), so we know the right answer.

https://academy.claude.com/courses/building-with-the-claude-api/image-support

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import base64
from pathlib import Path
from helpers.chat_helper import client, add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_usage


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


def image_block(image_bytes):
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_bytes,
        },
    }


def ask_about_image(image_bytes, prompt):
    """Sends the image and the prompt in one user message; returns the whole response"""
    messages = []
    add_user_message(messages, [
        image_block(image_bytes),            # Image block
        {"type": "text", "text": prompt},    # Text block
    ])
    response = send(messages, model=model, max_tokens=max_tokens)
    log_model_usage(response)
    return response


def image_tokens(image_bytes, prompt):
    """How many input tokens the image adds: count with and without it (free, no API answer)"""
    with_image = client.messages.count_tokens(model=model, messages=[
        {"role": "user", "content": [image_block(image_bytes), {"type": "text", "text": prompt}]},
    ])
    without_image = client.messages.count_tokens(model=model, messages=[
        {"role": "user", "content": prompt},
    ])
    return with_image.input_tokens - without_image.input_tokens


def main():
    # Read the file and encode it as base64 text.
    with open(image_file, "rb") as f:
        image_bytes = base64.standard_b64encode(f.read()).decode("utf-8")

    print(f"The image we send: {image_file}")

    log_banner("1. A BARE QUESTION")
    simple_response = ask_about_image(image_bytes, simple_prompt)

    log_banner("2. A STEP-BY-STEP PROMPT")
    step_by_step_response = ask_about_image(image_bytes, step_by_step_prompt)

    # Beyond the course: the token counting below is our addition.
    log_banner("3. BEYOND THE COURSE: WHAT THE IMAGE COSTS (count_tokens, free)")
    tokens_for_image = image_tokens(image_bytes, simple_prompt)
    print(f"The image adds {tokens_for_image} input tokens (estimate: 900 x 600 / 750 = 720).")

    print(f"\n=== ANSWER TO USER ===\nThe image has {actual_count} marbles.")
    print(f"\n--- bare question ---\n{text_from_message(simple_response)}")
    print(f"\n--- step-by-step prompt ---\n{text_from_message(step_by_step_response)}")

    print("\n=== TOKENS (beyond the course) ===")
    print(f"  {'prompt':16} {'input':>7} {'of which image':>15} {'output':>7}")
    for name, response in [("bare question", simple_response), ("step-by-step", step_by_step_response)]:
        usage = response.usage
        print(f"  {name:16} {usage.input_tokens:>7} {tokens_for_image:>15} {usage.output_tokens:>7}")
    print("\n  input includes the image; output includes thinking.")


if __name__ == "__main__":
    main()
