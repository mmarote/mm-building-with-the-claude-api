"""Building with the Claude API: Lesson 33 - Text chunking strategies

Before RAG can find the right part of a document, the document must be split into chunks, and
how it is split decides how good the answers can be; here we try three ways on the same report.

The strategies:

    size-based       fixed number of characters,     any text, even code.
                     with some overlap               Cuts words and sentences; a header can end up
                                                     in a different chunk than its content.
    structure-based  split at the document's         cleanest chunks: each one a whole section.
                     headers (## in Markdown)        Only works when the structure is guaranteed.
    sentence-based   groups of whole sentences,      a good middle ground for most text.
                     with some overlap
    semantic-based   group sentences by meaning      most relevant chunks, but slow and complex
                     (not shown here)                (it needs to understand each sentence).

Overlap: the end of one chunk is repeated at the start of the next, so an idea cut in two still
appears whole in at least one chunk.

Which to use (there is no single best one):
- You control the format (e.g. internal reports): structure-based.
- Most other text: sentence-based.
- Anything, as a reliable fallback (the usual choice in production): size-based with overlap.

The document: data/05-rag-and-agentic-search/report.md, a short fictional company report with one
"## " section per division. The course does not provide its file; this one is used again in the
next lessons.

No Claude API call: chunking is plain Python.

https://academy.claude.com/courses/building-with-the-claude-api/text-chunking-strategies
"""

import re
from pathlib import Path


# The report, in the data folder at the project root.
report_file = Path(__file__).resolve().parents[2] / "data" / "05-rag-and-agentic-search" / "report.md"


def chunk_by_char(text, chunk_size=150, chunk_overlap=20):
    """Size-based: chunks of chunk_size characters; each one repeats the last chunk_overlap characters"""
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunks.append(text[start_idx:end_idx])

        # Step back by the overlap for the next chunk, unless this was the last one.
        start_idx = end_idx - chunk_overlap if end_idx < len(text) else len(text)

    return chunks


def chunk_by_section(document_text):
    """Structure-based: one chunk per "## " section (the first chunk is the text before the first header)"""
    pattern = r"\n## "
    return re.split(pattern, document_text)


def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    """Sentence-based: groups of whole sentences; consecutive chunks share overlap_sentences sentences"""
    # A sentence ends at . ! or ? followed by whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))
        chunks.append(" ".join(sentences[start_idx:end_idx]))

        start_idx += max_sentences_per_chunk - overlap_sentences
        if start_idx < 0:
            start_idx = 0

    return chunks


def show(title, chunks, how_many=3):
    """Prints how many chunks a strategy made, their sizes, and the first few"""
    sizes = [len(chunk) for chunk in chunks]
    print(f"\n{'=' * 60}\n{title}: {len(chunks)} chunks, {min(sizes)}-{max(sizes)} characters\n{'=' * 60}")
    for number, chunk in enumerate(chunks[:how_many], start=1):
        print(f"\n--- chunk {number} ---\n{chunk.strip()}")


def main():
    text = report_file.read_text(encoding="utf-8")
    print(f"The report: {len(text)} characters ({report_file.name})")

    # Size-based: look at the start and end of each chunk; words and sentences are cut.
    show("SIZE-BASED (chunk_by_char, 150 characters, overlap 20)", chunk_by_char(text))

    # Structure-based: each chunk is a whole section (its "## " is removed by the split).
    show("STRUCTURE-BASED (chunk_by_section)", chunk_by_section(text))

    # Sentence-based: whole sentences; the last sentence of a chunk starts the next one.
    # The sentence rule knows nothing about headers, so "## Section ..." can land mid-chunk.
    show("SENTENCE-BASED (chunk_by_sentence, 5 sentences, overlap 1)", chunk_by_sentence(text))


if __name__ == "__main__":
    main()
