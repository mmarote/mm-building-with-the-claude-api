"""Building with the Claude API: Lesson 37 - BM25 lexical search

Semantic search can miss exact terms such as an incident ID, so here we add BM25, a lexical
search that ranks chunks by the words they share with the question, rare words counting
most.

The problem (from the course):

    question: "What happened with INC-2023-Q4-011?"

    semantic search  finds chunks that are ABOUT similar things, so it may return the finance
                     section, which never mentions the incident.
    lexical search   finds chunks that CONTAIN the same words, and "inc-2023-q4-011" appears
                     only in the cybersecurity section.

    Best of both: run them side by side and merge the results (lesson 38).

How BM25 scores a chunk:

    1. tokenize    "What happened with INC-2023-Q4-011?"
                   -> what | happened | with | inc-2023-q4-011
    2. count       how many chunks contain each term
                   "with" in most chunks, "inc-2023-q4-011" in one
    3. weight      rare term = high weight, common term = low weight
    4. rank        chunks with more of the heavily weighted terms come first
                   (a short chunk with a term counts a bit more than a long one)

Our additions (the course uses BM25Index but does not show its code):
- BM25Index with the standard BM25 formula, in plain Python (k1=1.5, b=0.75: the usual values).
- tokenize keeps hyphenated terms together, so INC-2023-Q4-011 and XR-472 stay one token.
- search returns (document, distance) pairs, lower = better, like VectorIndex in lesson 36:
  distance = 1 / (1 + score), so 1.0 means no query term matched.

No API call at all: BM25 is plain Python, so this lesson is free to run.

https://academy.claude.com/courses/building-with-the-claude-api/bm25-lexical-search

The log_* calls come from helpers/log_helper.py and only print what is happening;
they are not part of the lesson.
"""

import math
import re
from pathlib import Path
from helpers.log_helper import log_banner


# The report from lesson 33, in the data folder at the project root.
report_file = Path(__file__).resolve().parents[2] / "data" / "05-rag-and-agentic-search" / "report.md"

user_question = "What happened with INC-2023-Q4-011?"
how_many_results = 3


def chunk_by_section(document_text: str) -> list[str]:
    """Structure-based chunking from lesson 33: one chunk per "## " section"""
    pattern = r"\n## "
    return re.split(pattern, document_text)


def title_of(chunk: str) -> str:
    """The first line of a chunk (its section header), to name it in the output"""
    return chunk.strip().splitlines()[0]


def tokenize(text: str) -> list[str]:
    """Lowercase words; hyphenated terms such as INC-2023-Q4-011 stay one token"""
    return re.findall(r"\w+(?:-\w+)*", text.lower())


class BM25Index:
    """A tiny lexical search index: stores documents as words, ranks them with BM25"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1  # how quickly repeats of a term stop adding to the score
        self.b = b    # how much a long chunk is penalized (0 = not at all, 1 = fully)
        self.documents: list[dict] = []
        self.doc_tokens: list[list[str]] = []

    def add_document(self, document: dict) -> None:
        """Stores one document (a dict, e.g. {"content": chunk}) and its words"""
        self.documents.append(document)
        self.doc_tokens.append(tokenize(document["content"]))

    def chunks_containing(self, term: str) -> int:
        return sum(1 for tokens in self.doc_tokens if term in tokens)

    def term_weight(self, term: str) -> float:
        """Inverse document frequency: high for rare terms, near zero for terms in every document"""
        total = len(self.doc_tokens)
        containing = self.chunks_containing(term)
        return math.log((total - containing + 0.5) / (containing + 0.5) + 1)

    def score(self, query_terms: list[str], tokens: list[str]) -> float:
        """The BM25 score of one document: the weights of the query terms it contains"""
        average_length = sum(len(t) for t in self.doc_tokens) / len(self.doc_tokens)
        length_factor = 1 - self.b + self.b * len(tokens) / average_length
        total = 0.0
        for term in query_terms:
            count = tokens.count(term)
            if count:
                total += self.term_weight(term) * count * (self.k1 + 1) / (count + self.k1 * length_factor)
        return total

    def search(self, query: str, k: int = 1) -> list[tuple[dict, float]]:
        """Returns the k best documents as (document, distance) pairs, closest first"""
        query_terms = tokenize(query)
        scores = [self.score(query_terms, tokens) for tokens in self.doc_tokens]
        ranked = sorted(zip(self.documents, scores), key=lambda pair: pair[1], reverse=True)
        # A distance like VectorIndex's (lower = better): 1.0 when no query term matched.
        return [(doc, 1 / (1 + score)) for doc, score in ranked[:k]]


def main() -> None:
    # 1. Chunk the report by section.
    log_banner("STEP 1: CHUNK THE REPORT")
    text = report_file.read_text(encoding="utf-8")
    chunks = chunk_by_section(text)
    print(f"{report_file.name}: {len(text)} characters -> {len(chunks)} chunks")

    # 2. Create a BM25 store and add the chunks. No embeddings: BM25 only needs the words.
    log_banner("STEP 2: ADD THE CHUNKS (AS WORDS)")
    store = BM25Index()
    for number, chunk in enumerate(chunks, start=1):
        store.add_document({"content": chunk})
        tokens = store.doc_tokens[-1]
        print(f"  [{number}] {len(tokens):3} words  {title_of(chunk)}")
        print(f"      first words: {' | '.join(tokens[:8])} ...")

    # 3. Search. First, how BM25 sees the question: its terms and their weights.
    log_banner("STEP 3: LEXICAL SEARCH (BM25)")
    print(f"Question: {user_question}\n")
    # The weight only counts in chunks that contain the term: a term in no chunk adds nothing.
    print("  term               in chunks   weight (rarer = higher)")
    for term in tokenize(user_question):
        containing = store.chunks_containing(term)
        weight = f"{store.term_weight(term):5.2f}" if containing else "    -   (in no chunk: adds nothing)"
        print(f"  {term:18} {containing:4} of {len(chunks)}   {weight}")

    results = store.search(user_question, how_many_results)

    # The summary: what the search found.
    log_banner("SUMMARY: LEXICAL SEARCH RESULTS (BM25)")
    print(f"Question: {user_question}")
    print("distance: lower = closer match (1.000 = no query term found)")
    for rank, (doc, distance) in enumerate(results, start=1):
        best = "   <- best match" if rank == 1 else ""
        print(f"\n#{rank}  distance {distance:.3f}{best}\n{doc['content'][:200]}\n----")

    # Not in the course: every chunk, ranked, with its BM25 score (higher = better) and the
    # question's terms it contains, to see why it ranked where it did.
    print("\nAll chunks, closest first:")
    query_terms = tokenize(user_question)
    for rank, (doc, distance) in enumerate(store.search(user_question, len(chunks)), start=1):
        tokens = tokenize(doc["content"])
        score = store.score(query_terms, tokens)
        matched = ", ".join(term for term in query_terms if term in tokens) or "nothing"
        best = "   <- best match" if rank == 1 else ""
        print(f"  #{rank}  distance {distance:.3f}  (score {score:.2f})  "
              f"{title_of(doc['content']):48} matched: {matched}{best}")


if __name__ == "__main__":
    main()
