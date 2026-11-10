"""Building with the Claude API: Lesson 42 - Citations

With citations turned on, Claude points to the exact parts of your documents that support
each part of its answer, so a reader can check them.

Turning them on: two extra fields on the document block.

    {"type": "document",
     "source": {...},                      a PDF (lesson 41) or plain text
     "title": "report.pdf",                <- new
     "citations": {"enabled": True}}       <- new

The answer comes back in several text blocks; a block backed by the document carries citations:

    text block  "The incident was contained within four hours"
      citations: cited_text       the exact sentence from the document
                 document_index   which document (0 = the first one sent)
                 document_title   the title we gave it
                 where:  PDF  -> start_page_number, end_page_number   (type page_location)
                         text -> start_char_index, end_char_index     (type char_location)

Useful when readers must verify facts, the sources are authoritative, or you need to show where
an answer came from (e.g. a marker that shows the source on hover).

Where to look in the output:
- <- RESPONSE (the log): the raw reply, every text block and its citations as the API sends them.
- ANSWER TO USER: the same reply as an app would show it, with [n] markers and a source list.

This script asks the same question twice, to show the two kinds of citation location
(not to compare costs; that is lesson 41):
- the report as a PDF        -> citations point to pages               (page_location)
- the report as plain text   -> citations point to character positions (char_location)

The documents: data/06-features-of-claude/report.pdf (5 pages) and
data/05-rag-and-agentic-search/report.md, the fictional report of lessons 33-38.

https://academy.claude.com/courses/building-with-the-claude-api/citations

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import base64
from pathlib import Path
from helpers.chat_helper import add_user_message, send
from helpers.models import CHEAPEST
from helpers.log_helper import log_banner


data_dir = Path(__file__).resolve().parents[2] / "data"
pdf_file = data_dir / "06-features-of-claude" / "report.pdf"
text_file = data_dir / "05-rag-and-agentic-search" / "report.md"

model = CHEAPEST
# Thinking counts toward max_tokens; you only pay for the tokens Claude actually uses.
max_tokens = 16000
# Two facts from two different sections, so the answer needs at least two citations.
user_prompt = "What happened with incident INC-2023-Q4-011, and how much did the group spend on research and development?"


def pdf_document_block():
    with open(pdf_file, "rb") as f:
        file_bytes = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": file_bytes,
        },
        "title": "report.pdf",              # <- new
        "citations": {"enabled": True},     # <- new
    }


def text_document_block():
    return {
        "type": "document",
        "source": {
            "type": "text",                 # <- new: plain text, no base64
            "media_type": "text/plain",     # <- new
            "data": text_file.read_text(encoding="utf-8"),
        },
        "title": "report_text",             # <- new
        "citations": {"enabled": True},     # <- new
    }


def ask(document_block):
    messages = []
    add_user_message(messages, [document_block, {"type": "text", "text": user_prompt}])
    return send(messages, model=model, max_tokens=max_tokens)


def show(response):
    """Prints the answer with a [n] marker after each cited part, then the list of citations"""
    answer = []
    citations = []
    for block in response.content:
        if block.type == "text":
            text = block.text
            for citation in block.citations or []:
                citations.append(citation)
                text += f" [{len(citations)}]"
            answer.append(text)

    print("".join(answer))
    print("\nCitations:")
    for number, citation in enumerate(citations, start=1):
        if citation.type == "page_location":
            where = f"page {citation.start_page_number}"
            if citation.end_page_number - 1 > citation.start_page_number:
                where += f"-{citation.end_page_number - 1}"  # end_page_number is exclusive
        elif citation.type == "char_location":
            where = f"characters {citation.start_char_index}-{citation.end_char_index}"
        else:
            where = citation.type
        print(f"  [{number}] {citation.document_title}, {where}\n      \"{citation.cited_text.strip()}\"")


def main():
    log_banner("1. THE REPORT AS A PDF (citations point to pages)")
    pdf_response = ask(pdf_document_block())

    print("\n=== ANSWER TO USER ===")
    print("--- from the PDF (citations give pages) ---\n")
    show(pdf_response)


    log_banner("2. THE REPORT AS PLAIN TEXT (citations point to character positions)")
    text_response = ask(text_document_block())

    print("\n=== ANSWER TO USER ===")
    print("\n--- from the plain text (citations give character positions) ---\n")
    show(text_response)


if __name__ == "__main__":
    main()
