"""Building with the Claude API: Lesson 41 - PDF support

Claude can read PDF files: the code is the image code of lesson 40 with two changes, and here we ask for a one-sentence summary of a report.

From an image to a PDF (lesson 40 -> this lesson):

    "type": "image"          ->  "type": "document"
    "media_type": "image/png" ->  "media_type": "application/pdf"
    image_bytes              ->  file_bytes     (only the name, for clarity)

What Claude reads in a PDF: the text, and also images, charts, tables and the layout of each page.

What a PDF costs: each page is sent twice, as its extracted text and as an image of the page.

    PDF page   =  text tokens  +  image tokens      (the image is what shows charts and layout)
    plain text =  text tokens only

So the same report costs several times more as a PDF. If only the words matter, send text;
send a PDF when charts, tables or layout matter. This script compares both with
client.messages.count_tokens (free, no answer generated).

The PDF: data/06-features-of-claude/report.pdf, the fictional Northwind report of lessons 33-38
as a 5-page PDF (the course uses a Wikipedia article about Earth, which it does not provide).

https://academy.claude.com/courses/building-with-the-claude-api/pdf-support

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import base64
from pathlib import Path
from helpers.chat_helper import client, add_user_message, send, text_from_message
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner, log_model_usage


data_dir = Path(__file__).resolve().parents[2] / "data"
pdf_file = data_dir / "06-features-of-claude" / "report.pdf"
text_file = data_dir / "05-rag-and-agentic-search" / "report.md"  # the same report, as plain text
pdf_pages = 5

model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000
user_prompt = "Summarize the document in one sentence"


def count_input_tokens(content):
    """Input tokens for one user message with this content (free, no answer generated)"""
    return client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": content}]
    ).input_tokens


def main():
    with open(pdf_file, "rb") as f:
        file_bytes = base64.standard_b64encode(f.read()).decode("utf-8")

    pdf_block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": file_bytes,
        },
    }

    log_banner("1. SUMMARIZE THE PDF")
    messages = []
    add_user_message(messages, [pdf_block, {"type": "text", "text": user_prompt}])
    response = send(messages, model=model, max_tokens=max_tokens)
    log_model_usage(response)

    log_banner("2. THE SAME REPORT: PDF VS. PLAIN TEXT (count_tokens, free)")
    text_block = {"type": "text", "text": text_file.read_text(encoding="utf-8")}
    question = {"type": "text", "text": user_prompt}
    pdf_tokens = count_input_tokens([pdf_block, question])
    text_tokens = count_input_tokens([text_block, question])
    print(f"  {'format':12} {'input tokens':>13} {'per page':>9}")
    print(f"  {'PDF':12} {pdf_tokens:>13} {pdf_tokens // pdf_pages:>9}")
    print(f"  {'plain text':12} {text_tokens:>13} {text_tokens // pdf_pages:>9}")
    print(f"\n  The PDF costs {pdf_tokens - text_tokens} more tokens "
          f"({pdf_tokens / text_tokens:.1f}x): mostly the image of each page.")

    print(f"\n=== ANSWER TO USER ===\n{text_from_message(response)}")


if __name__ == "__main__":
    main()
