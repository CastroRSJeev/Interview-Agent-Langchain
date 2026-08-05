import base64
import json
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import fitz  
from dotenv import load_dotenv
import os
load_dotenv()

pdf_path = "./resumes/resume.pdf"


def pdf_to_base64_images(path, zoom=2.0):
    """Convert each PDF page into a base64-encoded JPEG string."""
    doc = fitz.open(path)
    mat = fitz.Matrix(zoom, zoom)  # zoom>1 = higher resolution, better OCR by the LLM
    images_b64 = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        images_b64.append(b64_str)
    doc.close()
    return images_b64


def build_content_payload(images_b64):
    """Build the multimodal message content: text prompt + page images."""
    content_payload = [
        {
            "type": "text",
            "text": (
                "This is a multi-page or a single-page resume provided in sequential order. "
                "Extract all details and synthesize them into two fields: 'description' and 'jobrole'. "
                "The description should be written considering projects, experience, and technical skills. "
                "The jobrole field should contain a suitable job title along with years of experience. "
                "Return ONLY valid JSON, with no markdown code fences and no extra commentary. "
                "Example: "
                '{"description": "I am a Full-Stack and AI Engineer with over six years of experience '
                "building fast web applications and intelligent AI tools using Python, TypeScript, React, "
                "Node.js, and SQL. Throughout my career, I have modernized enterprise platforms—like "
                "speeding up a web app for 150,000 daily users by 42%—and built production AI systems, "
                "including an AI document search tool that boosted retrieval accuracy by 28% using "
                "LangChain and Gemini. My key projects range from a real-time analytics engine processing "
                "10,000 requests per second to a multimodal AI assistant that extracts structured data "
                'directly from multi-page PDFs.", "jobrole": "Full Stack AI Engineer with 6 years of experience"}'
            ),
        }
    ]

    for b64_str in images_b64:
        content_payload.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"},
            }
        )

    return content_payload


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def parse_llm_json(raw_content):
    text = extract_text(raw_content).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _parse_resume() -> dict:
    images_b64 = pdf_to_base64_images(pdf_path)
    content_payload = build_content_payload(images_b64)
    llm = ChatGoogleGenerativeAI(model=os.getenv('GEMINI_MODEL'), google_api_key=os.getenv('GEMINI_API_KEY'))
    response = llm.invoke([HumanMessage(content=content_payload)])
    return parse_llm_json(response.content)


def parse_resume_bytes(pdf_bytes: bytes) -> dict:
    """Parse a resume from raw PDF bytes — used by the FastAPI endpoint."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        images_b64 = pdf_to_base64_images(tmp_path)
    finally:
        os.remove(tmp_path)
    content_payload = build_content_payload(images_b64)
    llm = ChatGoogleGenerativeAI(model=os.getenv('GEMINI_MODEL'), google_api_key=os.getenv('GEMINI_API_KEY'))
    response = llm.invoke([HumanMessage(content=content_payload)])
    return parse_llm_json(response.content)


def get_projects() -> str:
    return _parse_resume().get("description", "")


def get_jobrole() -> str:
    return _parse_resume().get("jobrole", "")


def main():
    images_b64 = pdf_to_base64_images(pdf_path)
    content_payload = build_content_payload(images_b64)

    llm = ChatGoogleGenerativeAI(model=os.getenv('GEMINI_MODEL'),google_api_key=os.getenv('GEMINI_API_KEY'))
    message = HumanMessage(content=content_payload)

    response = llm.invoke([message])

    try:
        result = parse_llm_json(response.content)
        print(json.dumps(result, indent=2))
    except (json.JSONDecodeError, TypeError) as e:
        print("Failed to parse JSON from model response:")
        print(response.content)
        raise e


if __name__ == "__main__":
    main()