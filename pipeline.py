"""The retrieval pipeline, complete and ready to run.

Nothing in here is left for you to fill in. The lab walks you through it one
step at a time, and you run each step on its own:

    python pipeline.py split                  # cut the corpus into chunks and print the first four
    python pipeline.py split 6                # print chunk 6 and the three after it
    python pipeline.py split-naive            # the same split with the separator list emptied
    python pipeline.py index                  # embed the chunks and store them (run once)
    python pipeline.py retrieve "a question"  # show the passages that come back
    python pipeline.py prompt "a question"    # print the prompt without calling the model
    python pipeline.py ask "a question"       # the whole thing, end to end

Run `index` once. It costs money, takes a couple of minutes, and drops whatever
was in the store before. `split`, `retrieve` and `ask` are cheap and you can run
them as often as you like.

Four values in this file are marked with a DECISION comment, because you get to
choose them and the choice has consequences. The lab shows you what each one does on the sample
corpus, and you choose them properly for Casebook this afternoon.
"""

import os
import re
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import observe


CORPUS_PATH = "sample_corpus/"
COLLECTION_NAME = "sample_docs"
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"

# DECISION: how much text goes in each chunk, and how much neighbouring chunks
# share. The chunking lab shows you what happens when you move them.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# The splitter tries these in order and only cuts at an arbitrary character
# position when nothing earlier in the list is available. On this corpus that
# list matters more than either number above. `split-naive` runs the same split
# with the list emptied, so you can see the difference.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# DECISION: how many passages to retrieve for each question.
TOP_K = 1

# DECISION: the three things this prompt insists on are why answers stay inside
# the corpus and arrive with a citation. The grounding lab takes them apart.
SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using only the passages provided below.
If the answer is not contained in the passages, say "I cannot find this in the available documents."
For each piece of information you use, cite the source document in the format [Source: filename].
Do not use your general knowledge. Do not speculate."""


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
          "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}


def _roman(s: str) -> int:
    return _ROMAN.get(s.upper(), 0)


def _breadcrumb(doc_name, chapter, chapter_title, section, section_title,
                article, article_title, paragraph, point=None) -> str:
    parts = [doc_name]
    if chapter:
        parts.append(f"Chapter {_ROMAN.get(str(chapter), chapter)}: {chapter_title}")
    if section:
        parts.append(f"Section {section}: {section_title}")
    if article:
        parts.append(f"Article {article}: {article_title}")
    if paragraph:
        label = f"Paragraph {paragraph}({point})" if point else f"Paragraph {paragraph}"
        parts.append(label)
    return " — ".join(parts)


def parse_structured_documents(corpus_path: str = CORPUS_PATH) -> tuple[list[Document], list[str]]:
    """Parse legal documents with chapter/section/article/paragraph/point structure.

    Expects headings on their own lines in one of these forms (adjust the regex
    patterns below to match your actual document format):
        CHAPTER III
        High-Risk AI Systems
        Section 2
        Requirements for high-risk AI systems
        Article 9
        Risk management system
        5. Paragraph text that may run over multiple lines.
        (a) Point text that may also run over multiple lines.

    Returns (documents, ids) — pass both to build_vector_store().
    """
    # ---- regex patterns: tune these to your document's formatting ----
    RE_CHAPTER  = re.compile(r"^CHAPTER\s+([IVXLC]+)\s*$", re.IGNORECASE)
    RE_SECTION  = re.compile(r"^Section\s+(\d+)\s*$", re.IGNORECASE)
    RE_ARTICLE  = re.compile(r"^Article\s+(\d+)\s*$", re.IGNORECASE)
    RE_PARA     = re.compile(r"^(\d+)\.\s+(.*)")
    RE_POINT    = re.compile(r"^\(([a-z])\)\s+(.*)")
    RE_ANNEX    = re.compile(r"^ANNEX\s+([IVXLC]+)\b", re.IGNORECASE)
    # ------------------------------------------------------------------

    docs: list[Document] = []
    ids:  list[str]      = []

    for path in sorted(Path(corpus_path).glob("*.txt")):
        doc_slug = path.stem                        # e.g. "eu-ai-act"
        doc_name = "EU AI Act"                      # override per file if needed
        lines    = path.read_text(encoding="utf-8").splitlines()

        chapter_num, chapter_title = None, ""
        section_num, section_title = None, ""
        article_num, article_title = None, ""
        annex_num,   annex_title   = None, ""

        def _collect(start: int, stops) -> tuple[str, int]:
            """Read continuation lines until a stop pattern matches."""
            text, i = lines[start], start + 1
            while i < len(lines):
                line = lines[i].strip()
                if any(p.match(line) for p in stops):
                    break
                if line:
                    text += " " + line
                i += 1
            return text.strip(), i

        stop_all = [RE_CHAPTER, RE_SECTION, RE_ARTICLE, RE_PARA, RE_POINT, RE_ANNEX]

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Preamble: nothing before the first CHAPTER is parsed into chunks,
            # because article_num is None until the first Article heading is seen.

            # Annexes: clear article context and start tracking annex state.
            m = RE_ANNEX.match(line)
            if m:
                chapter_num, chapter_title = None, ""
                section_num, section_title = None, ""
                article_num, article_title = None, ""
                annex_num   = _roman(m.group(1))
                annex_title = lines[i + 1].strip() if i + 1 < len(lines) else ""
                i += 2
                continue

            m = RE_CHAPTER.match(line)
            if m:
                chapter_num   = _roman(m.group(1))
                chapter_title = lines[i + 1].strip() if i + 1 < len(lines) else ""
                section_num, section_title = None, ""
                article_num, article_title = None, ""
                i += 2
                continue

            m = RE_SECTION.match(line)
            if m:
                section_num   = int(m.group(1))
                section_title = lines[i + 1].strip() if i + 1 < len(lines) else ""
                article_num, article_title = None, ""
                i += 2
                continue

            m = RE_ARTICLE.match(line)
            if m:
                article_num   = int(m.group(1))
                article_title = lines[i + 1].strip() if i + 1 < len(lines) else ""
                i += 2
                continue

            m = RE_PARA.match(line)
            if m and (article_num or annex_num):
                para_num  = int(m.group(1))
                para_text, i = _collect(i, stop_all)
                para_text = re.sub(r"^\d+\.\s+", "", para_text)

                # Collect any points that immediately follow
                points: list[tuple[str, str]] = []
                while i < len(lines):
                    pm = RE_POINT.match(lines[i].strip())
                    if not pm:
                        break
                    point_text, i = _collect(i, stop_all)
                    point_text = re.sub(r"^\([a-z]\)\s+", "", point_text)
                    points.append((pm.group(1), point_text))

                if annex_num:
                    # ---- annex chunk ----
                    annex_label = f"Annex {annex_num}: {annex_title}"
                    if points:
                        for letter, point_text in points:
                            chunk_id = f"{doc_slug}-annex{annex_num}-p{para_num}-{letter}"
                            ids.append(chunk_id)
                            docs.append(Document(
                                page_content=f"{doc_name} — {annex_label} — Paragraph {para_num}({letter}): {point_text}",
                                metadata={
                                    "document":      doc_name,
                                    "annex":         annex_num,
                                    "annex_title":   annex_title,
                                    "paragraph":     para_num,
                                    "point":         letter,
                                    "content_type":  "annex",
                                },
                            ))
                    else:
                        chunk_id = f"{doc_slug}-annex{annex_num}-p{para_num}"
                        ids.append(chunk_id)
                        docs.append(Document(
                            page_content=f"{doc_name} — {annex_label} — Paragraph {para_num}: {para_text}",
                            metadata={
                                "document":      doc_name,
                                "annex":         annex_num,
                                "annex_title":   annex_title,
                                "paragraph":     para_num,
                                "point":         None,
                                "content_type":  "annex",
                            },
                        ))
                else:
                    # ---- article chunk ----
                    if points:
                        for letter, point_text in points:
                            chunk_id = f"{doc_slug}-art{article_num}-p{para_num}-{letter}"
                            crumb    = _breadcrumb(doc_name, chapter_num, chapter_title,
                                                   section_num, section_title,
                                                   article_num, article_title,
                                                   para_num, letter)
                            ids.append(chunk_id)
                            docs.append(Document(
                                page_content=f"{crumb}: {point_text}",
                                metadata={
                                    "document":      doc_name,
                                    "chapter":       chapter_num,
                                    "section":       section_num,
                                    "article":       article_num,
                                    "paragraph":     para_num,
                                    "point":         letter,
                                    "article_title": article_title,
                                    "content_type":  "legal_requirement",
                                },
                            ))
                    else:
                        chunk_id = f"{doc_slug}-art{article_num}-p{para_num}"
                        crumb    = _breadcrumb(doc_name, chapter_num, chapter_title,
                                               section_num, section_title,
                                               article_num, article_title, para_num)
                        ids.append(chunk_id)
                        docs.append(Document(
                            page_content=f"{crumb}: {para_text}",
                            metadata={
                                "document":      doc_name,
                                "chapter":       chapter_num,
                                "section":       section_num,
                                "article":       article_num,
                                "paragraph":     para_num,
                                "point":         None,
                                "article_title": article_title,
                                "content_type":  "legal_requirement",
                            },
                        ))
                continue

            i += 1

    print(f"Parsed {len(docs)} structured chunks from {corpus_path}")
    return docs, ids


def load_documents(corpus_path: str = CORPUS_PATH):
    """Read every .txt file in corpus_path.

    Each file becomes a Document with the text in page_content and the file path
    under metadata["source"]. That source key is what makes a citation possible
    at the other end of the pipeline.

    DirectoryLoader returns files in whatever order the filesystem gives it, so
    they are sorted by path here. Without that, chunk numbers change between
    machines and nothing the lab says about a particular chunk would hold.
    """
    loader = DirectoryLoader(
        corpus_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = sorted(loader.load(), key=lambda d: d.metadata["source"])
    print(f"Loaded {len(documents)} documents")
    return documents


def split_documents(documents, separators=None):
    """Cut each document into chunks of at most CHUNK_SIZE characters.

    The separators are tried in order, so the splitter looks for a paragraph
    break first and only cuts at an arbitrary character position once nothing
    earlier in the list is available.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS if separators is None else separators,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks, ids=None):
    """Embed every chunk and store it in pgvector.

    One call embeds each chunk and writes a row per chunk, holding the text, its
    vector and its metadata, tagged with the collection named by
    COLLECTION_NAME. pre_delete_collection clears that collection first, so this
    always starts from a clean state.

    The rows live in langchain_pg_embedding and the collection itself is a row
    in langchain_pg_collection. There is no table called "sample_docs".

    Pass ids (a list of strings the same length as chunks) to use structured
    identifiers instead of auto-generated UUIDs.
    """
    embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection=os.environ["PG_CONNECTION_STRING"],
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True,
        ids=ids,
    )
    print(f"Stored {len(chunks)} chunks in collection {COLLECTION_NAME!r}")
    return vector_store


def load_vector_store():
    """Connect to a collection that has already been populated.

    This embeds nothing, so it is cheap. Use it for every query after the single
    run of build_vector_store.
    """
    embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
    return PGVector(
        embeddings=embeddings,
        connection=os.environ["PG_CONNECTION_STRING"],
        collection_name=COLLECTION_NAME,
    )

@observe()
def retrieve(query: str, vector_store, k: int = TOP_K) -> list[dict]:
    """Find the k passages closest in meaning to query.

    Returns a list of dicts, each holding:
      content  the passage text
      source   the file it came from, carried through from the loader
      score    cosine distance, so a smaller number means a closer match

    Week 10's agent calls this function and depends on all three keys being
    present, so keep the shape if you change the body.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(score, 4),
        }
        for doc, score in results
    ]


def build_prompt(question: str, passages: list[dict]) -> str:
    """Assemble the string the model actually sees.

    Each passage is numbered and labelled with its filename, so the model has
    something concrete to cite.
    """
    numbered = []
    for i, passage in enumerate(passages, 1):
        filename = passage["source"].split("/")[-1]
        numbered.append(f"[Passage {i} | Source: {filename}]\n{passage['content']}")
    context = "\n\n".join(numbered)
    return f"{SYSTEM_PROMPT}\n\nPassages:\n{context}\n\nQuestion: {question}"

@observe()
def generate(question: str, passages: list[dict]) -> str:
    """Send the assembled prompt to Gemini and return the answer text.

    temperature=0 removes sampling randomness, so when an answer changes you
    know the retrieval changed rather than the model rolling a different dice.
    """
    llm = ChatVertexAI(model_name=CHAT_MODEL, temperature=0)
    response = llm.invoke(build_prompt(question, passages))
    return response.content

@observe()
def ask(question: str, vector_store) -> dict:
    """Retrieve passages for question, then answer from them.

    Returns the question, the answer, the sources behind it, and the passage
    text the model saw. Day 3's evaluation harness reads all four.
    """
    passages = retrieve(question, vector_store)
    return {
        "question": question,
        "answer": generate(question, passages),
        "sources": [p["source"] for p in passages],
        "contexts": [p["content"] for p in passages],
    }


def cmd_split(separators=None, first=0, last=4):
    chunks = split_documents(load_documents(), separators=separators)
    for i in range(first, min(last, len(chunks))):
        print(f"\n--- chunk {i}, {len(chunks[i].page_content)} characters ---")
        print(chunks[i].page_content)
        print(f"[source: {chunks[i].metadata.get('source')}]")


def cmd_index(structured=False):
    if structured:
        docs, ids = parse_structured_documents()
        build_vector_store(docs, ids=ids)
    else:
        build_vector_store(split_documents(load_documents()))


def cmd_retrieve(question: str):
    for passage in retrieve(question, load_vector_store()):
        filename = passage["source"].split("/")[-1]
        print(f"\n{passage['score']}  {filename}")
        print(passage["content"][:300])


def cmd_ask(question: str):
    result = ask(question, load_vector_store())
    print("\n" + result["answer"])
    print("\nPassages consulted:")
    for source in result["sources"]:
        print("  ", source.split("/")[-1])


def cmd_prompt(question: str):
    """Print the prompt without calling the model, so you can read it."""
    passages = retrieve(question, load_vector_store())
    print("\n" + build_prompt(question, passages))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    question = sys.argv[2] if len(sys.argv) > 2 else ""
    needs_question = ("retrieve", "ask", "prompt")

    if command == "split":
        # An optional chunk number prints that chunk and the three after it.
        first = int(question) if question.isdigit() else 0
        cmd_split(first=first, last=first + 4)
    elif command == "split-naive":
        first = int(question) if question.isdigit() else 0
        cmd_split(separators=[""], first=first, last=first + 4)
    elif command == "index":
        cmd_index(structured="--structured" in sys.argv)
    elif command == "index-structured":
        cmd_index(structured=True)
    elif command in needs_question and not question:
        print(f'This one needs a question: python pipeline.py {command} "how do bats navigate"')
    elif command == "retrieve":
        cmd_retrieve(question)
    elif command == "ask":
        cmd_ask(question)
    elif command == "prompt":
        cmd_prompt(question)
    else:
        print(__doc__)
