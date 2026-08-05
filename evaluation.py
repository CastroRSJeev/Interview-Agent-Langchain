import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from utils import load_cached_vectorstore
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import os

# Deterministic — accuracy matters more than creativity here
# llm_evaluator = ChatOllama(
#     model="llama3.2:latest",
#     temperature=0,
#     base_url="http://localhost:11434"
# )
llm_groq_evaluator = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0,
    max_tokens=None,
    timeout=60,
    max_retries=2
)
llm_gemini_evaluator = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL', 'gemini-flash-latest'),
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0,
)

def invoke_evaluator_with_fallback(prompt_value):
    try:
        return llm_groq_evaluator.invoke(prompt_value)
    except Exception as e:
        if 'rate_limit' in str(e).lower() or '429' in str(e):
            print("Groq rate limit hit — falling back to Gemini.")
            return llm_gemini_evaluator.invoke(prompt_value)
        raise

# ---------------------------------------------------------------------------
# Non-answer detection (handled in code, never sent to the LLM — this is what
# prevents hallucinated factual_errors on blank/refusal answers)
# ---------------------------------------------------------------------------

IRRELEVANT_ANSWER_PATTERNS = [
    r"^\s*$",                          # blank
    r"^(i\s+)?don'?t\s+know",
    r"^(i\s+)?dobnt\s+reply",
    r"^skip$",
    r"^sorry$",
    r"^n/?a$",
    r"not\s+my\s+job",
    r"^idk$",
    r"^i\s+dont\s+know",
    r"^no[,\.!\s]",                    # "No why should I?", "No.", "No,"
    r"^why\s+should\s+i",
    r"^i\s+refuse",
    r"^i\s+won'?t",
    r"^none",
]


def is_blank_or_trivially_irrelevant(answer: str) -> bool:
    """Catch obviously blank/refusal answers in code, so the LLM never has to
    'evaluate' empty content."""
    stripped = answer.strip().lower()
    if len(stripped) == 0:
        return True
    for pattern in IRRELEVANT_ANSWER_PATTERNS:
        if re.match(pattern, stripped):
            return True
    return False


def build_non_answer_result() -> dict:
    """Deterministic, hallucination-free result for blank/refusal answers."""
    return {
        "technical_accuracy": 0,
        "completeness": 0,
        "clarity": 0,
        "factual_errors": [],
        "missing_points": ["The candidate did not attempt to answer the question."],
        "verdict": "Weak",
        "feedback": "No substantive answer was provided. Please attempt to address the question directly with relevant technical detail.",
    }


# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

evaluation_prompt = PromptTemplate(
    input_variables=["question", "answer", "grounding_context"],
    template="""You are a strict, accuracy-focused senior technical interviewer evaluating a candidate's spoken answer.

Interview question:
"{question}"

Candidate's answer:
"{answer}"

Background reference (for fact-checking ONLY — NOT a checklist of required topics):
\"\"\"
{grounding_context}
\"\"\"
IMPORTANT: The background reference is broad context about the technology. Use it ONLY to verify
whether specific claims the candidate made are factually correct or incorrect. Do NOT treat every
topic mentioned in the reference as something the candidate was required to cover. Do NOT surface
topics from the reference as missing points unless the question itself explicitly asks about them.

CRITICAL RULE: Only evaluate what the candidate ACTUALLY wrote. Never invent, assume, or reference
technical content that does not appear in the candidate's answer. If the answer is short, vague, or
dismissive, that is a completeness/relevance problem — do not fabricate factual_errors to fill the gap.

CRITICAL RULE: Do not flag spelling or grammar mistakes as factual_errors. Only flag claims that are
technically incorrect or contradicted by the reference material.

STEP 1 — Relevance check: if the answer does not attempt to address the technical substance of the
question at all (e.g. "not my job", "there will be no impact" with no elaboration, off-topic, or
dismissive), treat it as effectively a non-answer:
- technical_accuracy: 0, completeness: 0, clarity: 0
- factual_errors: empty list
- missing_points: list only what the QUESTION specifically asked for that was not addressed
- verdict: "Weak"
Skip Steps 2-4 below.

STEP 2 — Extract what the candidate explicitly said: read the candidate's answer carefully and
identify every distinct technical point, concept, or claim they actually stated. This is your
ground truth for what was covered. Do not infer or assume — only count what is literally present.

STEP 3 — If the candidate DID attempt a genuine technical answer, evaluate strictly using ONLY the
reference material and well-established technical facts. Do not give credit for confident-sounding
but incorrect statements. Flag any claim that contradicts or is unsupported by the reference material
as a factual error, quoting only the specific incorrect phrase — not the entire answer.

STEP 4 — Identify missing points: compare what the candidate explicitly said (from Step 2) against
what the QUESTION specifically asked for. A point is only "missing" if:
  (a) the question directly asks for it, AND
  (b) it is genuinely absent from the candidate's answer text identified in Step 2.
Do NOT add a point as missing if the candidate said it, even in different words or phrasing.
Do NOT add topics from the background reference that the question did not ask about.

Score each category from 0 (no answer / completely irrelevant) to 5 (excellent):
- technical_accuracy: Are the facts, terminology, and mechanisms described correctly?
- completeness: Does the answer sufficiently address what the question asks?
- clarity: Is the answer well-structured and understandable?

CONSISTENCY RULES (strictly enforced):
- If missing_points is non-empty, completeness MUST be 4 or lower.
- completeness: 5 means the answer fully addressed everything the question asked — only valid when missing_points is empty.
- Verdict must be consistent with scores: "Strong" only if all three scores are 4-5, "Weak" if any score is 1-2, "Adequate" otherwise.

Respond with ONLY valid JSON in exactly this format, no other text:
{{
  "technical_accuracy": <int 0-5>,
  "completeness": <int 0-5>,
  "clarity": <int 0-5>,
  "factual_errors": ["...", "..."],
  "missing_points": ["...", "..."],
  "verdict": "Strong" | "Adequate" | "Weak",
  "feedback": "..."
}}
"""
)

evaluation_chain = evaluation_prompt | invoke_evaluator_with_fallback | (lambda x: x.content)


def get_grounding_for_question(tech_name: str, question: str, k: int = 3, max_chars: int = 2000) -> str:
    """Retrieve chunks specifically relevant to THIS question, so the evaluator
    has targeted ground truth to check the answer against."""
    vectorstore = load_cached_vectorstore(tech_name)
    if vectorstore is None:
        return ""

    results = vectorstore.similarity_search(question, k=k)
    chunks = []
    total_len = 0
    for doc in results:
        chunk_text = f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        if total_len + len(chunk_text) > max_chars:
            break
        chunks.append(chunk_text)
        total_len += len(chunk_text)

    return "\n\n---\n\n".join(chunks)


def parse_evaluation_json(raw_output: str) -> dict:
    """Strip markdown fences if present, parse JSON, and enforce score/missing_points consistency."""
    cleaned = re.sub(r'```json|```', '', raw_output).strip()
    result = None
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if result is None:
        return {
            "technical_accuracy": None,
            "completeness": None,
            "clarity": None,
            "factual_errors": [],
            "missing_points": [],
            "verdict": "Unparseable",
            "feedback": f"Could not parse evaluator output: {raw_output[:300]}",
        }

    # Enforce: completeness=5 is only valid when missing_points is empty
    missing = result.get("missing_points") or []
    if missing and result.get("completeness") == 5:
        result["completeness"] = 4

    # Re-enforce verdict consistency after any score correction
    ta = result.get("technical_accuracy") or 0
    co = result.get("completeness") or 0
    cl = result.get("clarity") or 0
    if all(s >= 4 for s in (ta, co, cl)):
        result["verdict"] = "Strong"
    elif any(s <= 2 for s in (ta, co, cl)):
        result["verdict"] = "Weak"
    else:
        result["verdict"] = "Adequate"

    return result


def evaluate_answer(tech_name: str, question: str, answer: str) -> dict:
    """Evaluate one candidate answer against grounding material retrieved for that specific question."""
    if is_blank_or_trivially_irrelevant(answer):
        return build_non_answer_result()

    grounding_context = get_grounding_for_question(tech_name, question)
    if not grounding_context:
        grounding_context = "No reference material available — evaluate using general technical knowledge only."

    raw_result = evaluation_chain.invoke({
        "question": question,
        "answer": answer,
        "grounding_context": grounding_context,
    })

    return parse_evaluation_json(raw_result)


def evaluate_interview(tech_name: str, qa_pairs: list) -> list:
    """Evaluate a batch of (question, answer) pairs. qa_pairs = [{"question": ..., "answer": ...}, ...]"""
    results = []
    for i, pair in enumerate(qa_pairs, 1):
        print(f"Evaluating answer {i}/{len(qa_pairs)}...")
        evaluation = evaluate_answer(tech_name, pair["question"], pair["answer"])
        results.append({
            "question": pair["question"],
            "answer": pair["answer"],
            "evaluation": evaluation,
        })
    return results


def print_evaluation_report(results: list):
    """Pretty-print a full interview evaluation report."""
    print("\n" + "=" * 60)
    print("INTERVIEW EVALUATION REPORT")
    print("=" * 60)

    scores = []
    for i, r in enumerate(results, 1):
        ev = r["evaluation"]
        print(f"\nQ{i}: {r['question']}")
        print(f"Answer: {r['answer'][:150]}{'...' if len(r['answer']) > 150 else ''}")
        print(f"  Technical Accuracy: {ev.get('technical_accuracy')}/5")
        print(f"  Completeness:       {ev.get('completeness')}/5")
        print(f"  Clarity:            {ev.get('clarity')}/5")
        print(f"  Verdict:            {ev.get('verdict')}")
        if ev.get("factual_errors"):
            print(f"  Factual Errors:")
            for err in ev["factual_errors"]:
                print(f"    - {err}")
        if ev.get("missing_points"):
            print(f"  Missing Points:")
            for mp in ev["missing_points"]:
                print(f"    - {mp}")
        print(f"  Feedback: {ev.get('feedback')}")

        if ev.get("technical_accuracy") is not None:
            scores.append(ev["technical_accuracy"])

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n{'=' * 60}")
        print(f"Average Technical Accuracy: {avg:.2f}/5")
        print("=" * 60)


if __name__ == "__main__":
    tech_name = input("Technology (must match a cached tech, e.g. 'PyTorch'): ").strip()

    qa_pairs = []
    print("\nEnter question + candidate answer pairs. Type 'done' as the question to finish.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() == "done":
            break
        a = input("Candidate's answer: ").strip()
        qa_pairs.append({"question": q, "answer": a})

    if not qa_pairs:
        print("No Q&A pairs entered. Exiting.")
    else:
        results = evaluate_interview(tech_name, qa_pairs)
        print_evaluation_report(results)