import os
import re
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import embeddings, load_cached_vectorstore, save_cache
from evaluation import evaluate_interview, print_evaluation_report
from resume_parser import get_projects, get_jobrole

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

# Main model for question generation - keep some creativity
# llm_ollama = ChatOllama(
#     model="llama3.2:latest",
#     temperature=0.7,
#     base_url="http://localhost:11434"
# )
llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.7,
    max_tokens=None,
    timeout=60,
    max_retries=2
)
llm_gemini = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL', 'gemini-flash-latest'),
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.7,
)

def invoke_with_groq_fallback(prompt_value):
    try:
        return llm_groq.invoke(prompt_value)
    except Exception as e:
        if 'rate_limit' in str(e).lower() or '429' in str(e):
            print("Groq rate limit hit — falling back to Gemini.")
            return llm_gemini.invoke(prompt_value)
        raise
# Deterministic model for small extraction tasks (tech name, years of experience)
llm_extractor = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL', 'gemini-flash-latest'),
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0,
)

resume = {"projects": get_projects(), "jobrole": get_jobrole()}
projects = resume["projects"]
user_input = resume["jobrole"]
CORPUS_DIR = "corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)

MIN_CONTENT_LEN = 300


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_tech_name(user_input: str) -> str:
    extract_prompt = PromptTemplate(
        input_variables=["input"],
        template="""Extract ONLY the name of the technology, framework, or skill mentioned in this text.
Do not include years of experience, job titles, or any other words.

Text: "{input}"

Respond with ONLY the technology name, nothing else."""
    )
    chain = extract_prompt | llm_extractor | (lambda x: x.content)
    return chain.invoke({"input": user_input}).strip()


def extract_years_of_experience(user_input: str) -> int:
    extract_prompt = PromptTemplate(
        input_variables=["input"],
        template="""Extract ONLY the number of years of experience mentioned in this text.

Text: "{input}"

Respond with ONLY the number (digits only, e.g. "3"). If no experience is mentioned, respond with "0"."""
    )
    chain = extract_prompt | llm_extractor | (lambda x: x.content)
    raw = chain.invoke({"input": user_input}).strip()
    match = re.search(r'\d+', raw)
    return int(match.group()) if match else 0


def get_experience_band(years: int) -> dict:
    if years <= 2:
        return {
            "label": "Junior (0-2 years)",
            "guidance": (
                "ALL 5 questions must be fundamentals-level only. Ask about basic syntax, "
                "core concepts, definitions, and simple usage. Do NOT ask about internals, "
                "distributed systems, scalability, production debugging, or architecture "
                "trade-offs — these are too advanced for this level."
            ),
        }
    elif years <= 5:
        return {
            "label": "Mid-level (3-5 years)",
            "guidance": (
                "ALL 5 questions must focus on practical implementation: debugging real issues, "
                "applying best practices, comparing approaches, and hands-on trade-offs. "
                "Avoid pure fundamentals ('what is X') and avoid deep architecture/scalability "
                "questions meant for senior/staff engineers."
            ),
        }
    else:
        return {
            "label": "Senior (6+ years)",
            "guidance": (
                "ALL 5 questions must be advanced: system architecture, scalability, "
                "performance at scale, design trade-offs, and edge-case/production scenarios. "
                "Avoid basic fundamentals entirely — assume mastery of the basics."
            ),
        }


# ---------------------------------------------------------------------------
# Web search + corpus extraction
# ---------------------------------------------------------------------------

def get_tavily_context(tech_name: str, max_results: int = 5):
    try:
        response = tavily_client.search(
            query=f"{tech_name} core concepts architecture how it works features explained",
            search_depth="advanced",
            max_results=max_results,
        )
    except Exception as e:
        print(f"Tavily search failed: {e}")
        return []

    results = response.get("results", [])
    return [
        {"title": r.get("title", "Untitled"), "url": r.get("url", "")}
        for r in results if r.get("url")
    ]


def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if len(line) > 25 or line.endswith(('.', '?', ':'))]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_corpus_and_retriever(sources: list, tech_name: str):
    if not sources:
        return None

    all_chunks = []
    corpus_parts = []
    source_urls = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for s in sources:
        url = s.get("url")
        title = s.get("title", "Untitled")
        if not url:
            continue
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            raw_text = "\n\n".join(doc.page_content for doc in docs)
            cleaned = clean_text(raw_text)

            if len(cleaned) < MIN_CONTENT_LEN:
                print(f"  Skipped (too thin, likely nav/listing page): {url}")
                continue

            corpus_parts.append(f"===== SOURCE: {title} =====\nURL: {url}\n\n{cleaned}\n")
            source_urls.append(url)

            chunks = splitter.split_text(cleaned)
            for chunk in chunks:
                all_chunks.append({"content": chunk, "source": url, "title": title})

            print(f"  Extracted usable content from: {url} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"  Failed to load {url}: {e}")

    if not corpus_parts:
        print("No usable content extracted from any source.")
        return None

    full_corpus_text = "\n\n".join(corpus_parts)

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', tech_name.strip())[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(CORPUS_DIR, f"{safe_name}_{timestamp}.txt")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write(full_corpus_text)
    print(f"Full corpus saved to: {snapshot_path}")

    texts = [c["content"] for c in all_chunks]
    metadatas = [{"source": c["source"], "title": c["title"]} for c in all_chunks]
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    save_cache(tech_name, vectorstore, full_corpus_text, source_urls)

    return vectorstore


def get_vectorstore_for_tech(tech_name: str):
    vectorstore = load_cached_vectorstore(tech_name)
    if vectorstore is not None:
        return vectorstore

    sources = get_tavily_context(tech_name)
    return build_corpus_and_retriever(sources, tech_name)


def retrieve_relevant_context(vectorstore, tech_name: str, k: int = 4, max_chars: int = 3000) -> str:
    if vectorstore is None:
        return ""

    query = f"{tech_name} key concepts, best practices, common pitfalls, and important recent features"
    results = vectorstore.similarity_search(query, k=k)

    context_chunks = []
    total_len = 0
    for doc in results:
        chunk_text = f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        if total_len + len(chunk_text) > max_chars:
            break
        context_chunks.append(chunk_text)
        total_len += len(chunk_text)

    return "\n\n---\n\n".join(context_chunks)


def parse_questions(raw_text: str) -> list:
    """Split the numbered question list output into a clean list of question strings."""
    lines = raw_text.strip().split('\n')
    questions = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if match:
            questions.append(match.group(1).strip())
    return questions if questions else [raw_text.strip()]


# ---------------------------------------------------------------------------
# Prompts / chains
# ---------------------------------------------------------------------------

question_generation_prompt = PromptTemplate(
    input_variables=['tech_name', 'experience_label', 'experience_guidance', 'projects', 'web_context'],
    template="""You are a senior technical lead preparing to interview a candidate.

Technology: {tech_name}
Candidate level: {experience_label}

Difficulty rule for this candidate (follow strictly):
{experience_guidance}

Candidate's relevant project experience:
\"\"\"
{projects}
\"\"\"

Up-to-date reference material on this technology:
\"\"\"
{web_context}
\"\"\"

Instructions:
- Generate exactly 5 interview questions on {tech_name}, following the difficulty rule above exactly.
- Of the 5 questions, include 2-3 that are directly grounded in the candidate's project experience above, IF the projects are relevant to {tech_name}. Otherwise generate general questions consistent with the difficulty rule.
- Use the reference material to ground questions in real, current concepts — do not fabricate details not supported by it.
- Avoid vague questions like "What is X?"
- Each question should require at least 2-3 sentences to answer well — avoid one-line factual questions.
- Number the questions 1 to 5.

Output only the questions — no extra commentary, preamble, or explanation.
"""
)
question_generation_chain = question_generation_prompt | invoke_with_groq_fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

tech_name = extract_tech_name(user_input)
years = extract_years_of_experience(user_input)
band = get_experience_band(years)
print(f"Detected technology: {tech_name} | Experience: {years} years -> {band['label']}")

vectorstore = get_vectorstore_for_tech(tech_name)
web_context = retrieve_relevant_context(vectorstore, tech_name, k=4, max_chars=3000)

result = question_generation_chain.invoke({
    "tech_name": tech_name,
    "experience_label": band["label"],
    "experience_guidance": band["guidance"],
    "projects": projects,
    "web_context": web_context if web_context else "No additional reference material available.",
})

print("\n" + result.content)
questions = parse_questions(result.content)

_INJECTION_PATTERN = re.compile(
    r"(ignore (previous|all|above)|disregard|forget (previous|all)|new instruction|"
    r"you are now|act as|pretend (you are|to be)|jailbreak|"
    r"<script|javascript:|on\w+\s*=|eval\(|exec\()",
    re.IGNORECASE,
)
MAX_ANSWER_LENGTH = 2000


def validate_answer_cli(answer: str) -> str | None:
    if not answer or not answer.strip():
        print("  [!] Answer cannot be empty. Skipping.")
        return None
    if len(answer) > MAX_ANSWER_LENGTH:
        print(f"  [!] Answer exceeds {MAX_ANSWER_LENGTH} characters. Skipping.")
        return None
    if _INJECTION_PATTERN.search(answer):
        print("  [!] Answer contains disallowed content. Skipping.")
        return None
    return answer.strip()


print("\nEnter your answer for each question (or type 'skip' to skip).\n")
qa_pairs = []
for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    answer = input("Your answer: ").strip()
    if answer.lower() == "skip":
        continue
    clean = validate_answer_cli(answer)
    if clean:
        qa_pairs.append({"question": q, "answer": clean})

if qa_pairs:
    results = evaluate_interview(tech_name, qa_pairs)
    print_evaluation_report(results)
else:
    print("No answers collected — skipping evaluation.")