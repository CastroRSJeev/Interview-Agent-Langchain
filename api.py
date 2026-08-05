import os
import re
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from utils import embeddings, load_cached_vectorstore, save_cache
from evaluation import evaluate_answer
from resume_parser import parse_resume_bytes
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=None,
    timeout=60,
    max_retries=2,
)
llm_gemini = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
)
llm_extractor = llm_gemini

CORPUS_DIR = "corpus"
MIN_CONTENT_LEN = 300
os.makedirs(CORPUS_DIR, exist_ok=True)


def invoke_with_groq_fallback(prompt_value):
    try:
        return llm_groq.invoke(prompt_value)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return llm_gemini.invoke(prompt_value)
        raise


# ---------------------------------------------------------------------------
# Helpers (sync — run in threadpool via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _get_content(msg) -> str:
    c = msg.content
    return (c[0].get("text", "") if isinstance(c, list) and c else c) or ""


def _extract_tech_name(user_input: str) -> str:
    prompt = PromptTemplate(
        input_variables=["input"],
        template='Extract ONLY the technology/framework/skill name from: "{input}". Respond with ONLY the name.',
    )
    return (prompt | llm_extractor | _get_content).invoke({"input": user_input}).strip()


def _extract_years(user_input: str) -> int:
    prompt = PromptTemplate(
        input_variables=["input"],
        template='Extract ONLY the number of years of experience from: "{input}". Respond with digits only. If none, respond "0".',
    )
    raw = (prompt | llm_extractor | _get_content).invoke({"input": user_input}).strip()
    match = re.search(r"\d+", raw)
    return int(match.group()) if match else 0


def _get_experience_band(years: int) -> dict:
    if years <= 2:
        return {
            "label": "Junior (0-2 years)",
            "guidance": (
                "ALL 5 questions must be fundamentals-level only. Ask about basic syntax, "
                "core concepts, definitions, and simple usage. Do NOT ask about internals, "
                "distributed systems, scalability, production debugging, or architecture trade-offs."
            ),
        }
    elif years <= 5:
        return {
            "label": "Mid-level (3-5 years)",
            "guidance": (
                "ALL 5 questions must focus on practical implementation: debugging real issues, "
                "applying best practices, comparing approaches, and hands-on trade-offs."
            ),
        }
    else:
        return {
            "label": "Senior (6+ years)",
            "guidance": (
                "ALL 5 questions must be advanced: system architecture, scalability, "
                "performance at scale, design trade-offs, and edge-case/production scenarios."
            ),
        }


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [l.strip() for l in text.split("\n")]
    lines = [l for l in lines if len(l) > 25 or l.endswith((".", "?", ":"))]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _build_corpus(sources: list, tech_name: str):
    if not sources:
        return None
    all_chunks, corpus_parts, source_urls = [], [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for s in sources:
        url, title = s.get("url"), s.get("title", "Untitled")
        if not url:
            continue
        try:
            docs = WebBaseLoader(url).load()
            cleaned = _clean_text("\n\n".join(d.page_content for d in docs))
            if len(cleaned) < MIN_CONTENT_LEN:
                continue
            corpus_parts.append(f"===== SOURCE: {title} =====\nURL: {url}\n\n{cleaned}\n")
            source_urls.append(url)
            for chunk in splitter.split_text(cleaned):
                all_chunks.append({"content": chunk, "source": url, "title": title})
        except Exception:
            pass
    if not corpus_parts:
        return None
    full_corpus = "\n\n".join(corpus_parts)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", tech_name.strip())[:50]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(CORPUS_DIR, f"{safe}_{ts}.txt"), "w", encoding="utf-8") as f:
        f.write(full_corpus)
    texts = [c["content"] for c in all_chunks]
    metas = [{"source": c["source"], "title": c["title"]} for c in all_chunks]
    vs = FAISS.from_texts(texts, embedding=embeddings, metadatas=metas)
    save_cache(tech_name, vs, full_corpus, source_urls)
    return vs


def _get_vectorstore(tech_name: str):
    vs = load_cached_vectorstore(tech_name)
    if vs:
        return vs
    try:
        resp = tavily_client.search(
            query=f"{tech_name} core concepts architecture how it works features explained",
            search_depth="advanced",
            max_results=5,
        )
        sources = [{"title": r.get("title", ""), "url": r.get("url", "")} for r in resp.get("results", []) if r.get("url")]
    except Exception:
        sources = []
    return _build_corpus(sources, tech_name)


def _retrieve_context(vs, tech_name: str, k: int = 4, max_chars: int = 3000) -> str:
    if vs is None:
        return ""
    query = f"{tech_name} key concepts, best practices, common pitfalls, and important recent features"
    results = vs.similarity_search(query, k=k)
    chunks, total = [], 0
    for doc in results:
        chunk = f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(chunks)


def _parse_questions(raw: str) -> list:
    questions = []
    for line in raw.strip().split("\n"):
        m = re.match(r"^\d+[.)]\s*(.+)", line.strip())
        if m:
            questions.append(m.group(1).strip())
    return questions or [raw.strip()]


question_generation_prompt = PromptTemplate(
    input_variables=["tech_name", "experience_label", "experience_guidance", "projects", "web_context"],
    template="""You are a senior technical lead preparing to interview a candidate.

Technology: {tech_name}
Candidate level: {experience_label}

Difficulty rule (follow strictly):
{experience_guidance}

Candidate's project experience:
\"\"\"{projects}\"\"\"

Reference material:
\"\"\"{web_context}\"\"\"

Instructions:
- Generate exactly 5 interview questions on {tech_name}, following the difficulty rule.
- Include 2-3 questions grounded in the candidate's project experience if relevant.
- Each question should require at least 2-3 sentences to answer.
- Number the questions 1 to 5.
- Output only the questions, no commentary.
""",
)


def _generate_questions_sync(tech_name: str, band: dict, projects: str, web_context: str) -> list:
    result = invoke_with_groq_fallback(
        question_generation_prompt.format(
            tech_name=tech_name,
            experience_label=band["label"],
            experience_guidance=band["guidance"],
            projects=projects,
            web_context=web_context or "No additional reference material available.",
        )
    )
    return _parse_questions(result.content)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("resumes", exist_ok=True)
    yield

app = FastAPI(title="Interview Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------

_INJECTION_PATTERN = re.compile(
    r"(ignore (previous|all|above)|disregard|forget (previous|all)|new instruction|"
    r"you are now|act as|pretend (you are|to be)|jailbreak|"
    r"<script|javascript:|on\w+\s*=|eval\(|exec\()",
    re.IGNORECASE,
)

MAX_ANSWER_LENGTH = 2000


def validate_answer(answer: str) -> str:
    if not answer or not answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")
    if len(answer) > MAX_ANSWER_LENGTH:
        raise HTTPException(status_code=400, detail=f"Answer exceeds {MAX_ANSWER_LENGTH} character limit.")
    if _INJECTION_PATTERN.search(answer):
        raise HTTPException(status_code=400, detail="Answer contains disallowed content.")
    return answer.strip()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnswerRequest(BaseModel):
    tech_name: str
    question: str
    answer: str


class EvaluateAllRequest(BaseModel):
    tech_name: str
    qa_pairs: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    pdf_bytes = await file.read()
    try:
        resume = await asyncio.to_thread(parse_resume_bytes, pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {e}")

    jobrole = resume.get("jobrole", "")
    projects = resume.get("description", "")

    tech_name, years = await asyncio.gather(
        asyncio.to_thread(_extract_tech_name, jobrole),
        asyncio.to_thread(_extract_years, jobrole),
    )
    band = _get_experience_band(years)

    return {
        "tech_name": tech_name,
        "years": years,
        "experience_label": band["label"],
        "projects": projects,
        "jobrole": jobrole,
    }


@app.post("/generate-questions")
async def generate_questions(payload: dict):
    tech_name = payload.get("tech_name", "")
    projects = payload.get("projects", "")
    years = int(payload.get("years", 0))

    if not tech_name:
        raise HTTPException(status_code=400, detail="tech_name is required.")

    band = _get_experience_band(years)

    vs, = await asyncio.gather(asyncio.to_thread(_get_vectorstore, tech_name))
    web_context = await asyncio.to_thread(_retrieve_context, vs, tech_name)

    questions = await asyncio.to_thread(_generate_questions_sync, tech_name, band, projects, web_context)

    return {"questions": questions, "tech_name": tech_name, "experience_label": band["label"]}


@app.post("/submit-answer")
async def submit_answer(req: AnswerRequest):
    answer = validate_answer(req.answer)
    evaluation = await asyncio.to_thread(evaluate_answer, req.tech_name, req.question, answer)
    return {"question": req.question, "answer": answer, "evaluation": evaluation}


@app.post("/evaluate-all")
async def evaluate_all(req: EvaluateAllRequest):
    tasks = [
        asyncio.to_thread(evaluate_answer, req.tech_name, pair["question"], validate_answer(pair["answer"]))
        for pair in req.qa_pairs
    ]
    evaluations = await asyncio.gather(*tasks)
    results = [
        {"question": p["question"], "answer": p["answer"], "evaluation": ev}
        for p, ev in zip(req.qa_pairs, evaluations)
    ]
    scores = [r["evaluation"].get("technical_accuracy", 0) or 0 for r in results]
    avg = sum(scores) / len(scores) if scores else 0
    return {"results": results, "average_technical_accuracy": round(avg, 2)}
