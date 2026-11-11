"""Building with the Claude API: Lesson 62 - Parallelization workflows

Instead of one long prompt that weighs every material at once, we send one focused request
per material at the same time, then one request that compares the results and picks one.

The workflow:

                        ┌─► metal    ─┐
    part description ───┼─► polymer  ─┼──► AGGREGATOR (Claude) ──► recommendation
     (same input to     └─► wood     ─┘     compares the analyses
      every request)
                     one request per material, all at once,
                     each with the criteria of ONE material

Why split it:
- Each request has one job, so Claude can focus on one material's criteria.
- Each prompt can be tested and improved on its own; a new material is one more request.
- The requests are independent, so they can run at the same time: the wait is about as long
  as the slowest request, not the sum of all of them.

How they run at the same time: asyncio, with the SDK's async client.

    async def analyze(material)        one request; "await" waits for Claude's answer
                                       and lets the other requests run meanwhile
    asyncio.gather(analyze(m), ...)    starts them all at once, waits for all of them,
                                       and returns the results in the order given
    asyncio.run(analyze_all())         runs the async part from our normal main()

Only the parallel part is async; the aggregator is an ordinary call with send(). To keep
the log readable, the requests are logged before they start and the responses after all
of them are back, in a fixed order.

The course vs. this script:
- The course's app sends an image of the part with each request, and shows the flow only
  as a diagram, with no criteria and no code. We describe the part in text, and wrote the
  criteria for each material ourselves.
- The course's app compares six materials (metal, polymer, ceramic, composite, elastomer,
  wood). We use three, to keep the cost of a run down: the workflow is the same.

https://academy.claude.com/courses/building-with-the-claude-api/parallelization-workflows

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import asyncio
import re
from anthropic.types import Message
import helpers.chat_helper as chat_helper
from helpers.chat_helper import add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_request, log_model_response, Stopwatch


model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000

part = """A wall bracket that holds a 2 kg outdoor security camera on a brick wall. L-shaped,
about 12 cm per side, with two screw holes. Outdoors all year: rain, direct sun, and
temperatures from -20 to 45 °C. Made in batches of about 5,000 per year."""

# One set of criteria per material: each parallel request gets only its own.
# Add a material here and the workflow sends one more request in parallel.
criteria = {
    "metal": "strength and stiffness under load, corrosion in rain (and how to prevent it), "
             "cost of machining or stamping",
    "polymer": "creep under a constant load, UV degradation in sun, brittleness at -20 °C, "
               "cost of injection molding",
    "wood": "rot and swelling outdoors, strength across the grain at the screw holes, "
            "maintenance",
}


def analysis_prompt(material: str) -> str:
    return f"""Evaluate how suitable {material} is for this part.

<part>
{part}
</part>

Consider only these criteria for {material}: {criteria[material]}.

Answer in at most five short bullet points, then a last line exactly like this:
Suitability: N/10"""


async def analyze(material: str) -> tuple[str, Message, float]:
    """One parallel request: the analysis of one material.
    Returns (material, response, seconds)"""
    messages = [{"role": "user", "content": analysis_prompt(material)}]
    with Stopwatch() as sw:
        response = await chat_helper.async_client.messages.create(          # <- new
            model=model, max_tokens=max_tokens, messages=messages,
        )
    return material, response, sw.seconds


async def analyze_all() -> list[tuple[str, Message, float]]:
    """Starts one request per material at the same time and waits for all of them"""
    return await asyncio.gather(*(analyze(material) for material in criteria))   # <- new


def suitability(text: str) -> str:
    match = re.search(r"Suitability:\s*(\d+)\s*/\s*10", text)
    return f"{match.group(1)}/10" if match else "?"


def main():
    log_banner("1. ONE REQUEST PER MATERIAL, ALL AT ONCE")
    for material in criteria:
        log_model_request(model=model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": analysis_prompt(material)}])

    # They all run at the same time; gather() returns the results in the order of criteria.
    with Stopwatch() as wall:
        results = asyncio.run(analyze_all())                                # <- new

    log_banner("2. THE ANSWERS (logged after all of them are back)")
    for material, response, seconds in results:
        print(f"\n--- {material} ({seconds:.1f}s) ---")
        log_model_response(response)

    log_banner("3. THE AGGREGATOR: COMPARE THE ANALYSES AND RECOMMEND ONE")
    analyses = "\n".join(
        f'<analysis material="{material}">\n{text_from_message(response)}\n</analysis>'
        for material, response, _ in results
    )
    messages = []
    add_user_message(messages, f"""Several experts each analyzed one material for this part.

<part>
{part}
</part>

{analyses}

Compare the analyses and recommend the best material for this part. Name the runner-up and
say in two or three sentences what decided it.""")                                 # <- new
    recommendation = text_from_message(send(messages, model=model, max_tokens=max_tokens))

    print("\n=== THE ANALYSES ===")
    for material, response, seconds in results:
        score = suitability(text_from_message(response))
        print(f"  {material:10} {score:>5}   ({seconds:.1f}s)")
    total = sum(seconds for _, _, seconds in results)
    print(f"\n  One after another they would take about {total:.1f}s; "
          f"in parallel they took {wall.seconds:.1f}s.")

    print(f"\n=== ANSWER TO USER ===\n{recommendation}")


if __name__ == "__main__":
    main()
