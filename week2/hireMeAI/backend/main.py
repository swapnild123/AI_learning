import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pypdf import PdfReader
from pydantic import BaseModel, Field



load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key)

model = "qwen/qwen3.6-27b"

app = FastAPI()



class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str



def read_pdf(file_path: Path) -> str:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Resume PDF not found: {file_path.resolve()}"
        )

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise ValueError(
            "No text could be extracted from the resume PDF."
        )

    return text



def parse_resume(resume_text: str) -> Resume:

    resume_schema = Resume.model_json_schema()

    system_prompt = f"""
You are a resume information extractor.

Extract information from the resume and return ONLY valid JSON.

Do not return:
- reasoning
- explanations
- markdown
- code fences
- <think> tags

Use exactly this schema:

{json.dumps(resume_schema)}

Rules:

1. Extract only information actually present in the resume.
2. Never invent information.
3. Missing single values must be null.
4. Missing lists must be [].
5. Keep descriptions short.
6. Keep project information concise.
7. Return the JSON object directly.
"""

    user_prompt = f"""
Extract the candidate information from this resume:

{resume_text}
"""

    try:
        response = client.chat.completions.create(
            model=model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            # IMPORTANT:
            # Qwen 3.6 was spending the output budget on <think>.
            # Disable reasoning for this extraction task.
            reasoning_effort="none",

            # Force valid JSON.
            response_format={
                "type": "json_object"
            },

            # Enough room for the structured resume.
            max_completion_tokens=1000,

            temperature=0
        )

    except Exception as e:
        raise RuntimeError(
            f"Groq resume parsing failed: {e}"
        ) from e

    raw_output = response.choices[0].message.content

    if not raw_output:
        raise ValueError(
            "Resume parser returned an empty response."
        )

    print("\n========== PARSED RESUME JSON ==========")
    print(raw_output)
    print("=========================================\n")

   
    try:
        data = json.loads(raw_output)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Groq returned invalid JSON:\n{raw_output}"
        ) from e

    
    try:
        resume = Resume(**data)

    except Exception as e:
        raise ValueError(
            f"Resume validation failed:\n{data}"
        ) from e

    return resume



def ask_candidate(
    question: str,
    resume: Resume
) -> str:

    resume_data = resume.model_dump_json(
        indent=2
    )

    system_prompt = f"""
You are an AI assistant representing a job candidate.

Here is the candidate's structured resume:

{resume_data}

Rules:

1. Answer only using information present in the resume.
2. Never invent information.
3. If the resume does not contain the answer, say:
"I don't have enough information to answer that."
4. Answer briefly and directly.
5. Be professional.
"""

    try:

        response = client.chat.completions.create(
            model=model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            # Chat does not need reasoning either.
            reasoning_effort="none",

            max_completion_tokens=200,

            temperature=0
        )

    except Exception as e:
        raise RuntimeError(
            f"Groq candidate chat failed: {e}"
        ) from e

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError(
            "Candidate chatbot returned an empty response."
        )

    return answer


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "hireMeAI is running"
    }



@app.post("/chat")
def chat(request: ChatRequest):

    try:

        # main.py is inside backend/
        # PDF is one level above backend/

        resume_path = Path(
            "../swapnildwivedi.resume.pdf"
        )

        # PDF → text
        resume_text = read_pdf(
            resume_path
        )

        # Text → Pydantic Resume
        resume = parse_resume(
            resume_text
        )

        # Resume → AI answer
        answer = ask_candidate(
            question=request.question,
            resume=resume
        )

        return {
            "answer": answer
        }

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except ValueError as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {e}"
        )