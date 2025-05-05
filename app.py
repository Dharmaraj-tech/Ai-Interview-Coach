from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
import fitz  # PyMuPDF for PDF parsing
import docx   # python-docx for DOCX parsing
from langchain.chat_models import ChatGooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- LangChain Setup -----
llm = ChatGooglePalm(temperature=0.7)

question_prompt = PromptTemplate.from_template(
    """
    You are an AI interview coach. Generate 10 diverse interview questions for the role of {job_role}.
    The categories should include technical, behavioral, HR/general, and situational.
    Use the resume content below if provided to customize the questions.

    Resume:
    {resume_text}

    Provide the output as a list of questions only.
    """
)
question_chain = LLMChain(llm=llm, prompt=question_prompt)

eval_prompt = PromptTemplate.from_template(
    """
    You are an AI evaluator. Score the following interview response:

    Question: {question}
    Answer: {answer}

    Provide:
    - Confidence score (0–10)
    - Clarity score (0–10)
    - Relevance score (0–10)
    - Feedback
    - Expected/model answer
    """
)
eval_chain = LLMChain(llm=llm, prompt=eval_prompt)

# ----- Data Models -----
class Question(BaseModel):
    id: str
    text: str
    category: str

class AnswerInput(BaseModel):
    question_id: str
    question: str
    answer: str

class EvaluationResult(BaseModel):
    question_id: str
    confidence: float
    clarity: float
    relevance: float
    feedback: str
    expected_answer: str

class EvaluationReport(BaseModel):
    user_id: str
    evaluations: List[EvaluationResult]
    average_score: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_plan: str

# ----- In-Memory Stores (for prototyping) -----
QUESTIONS_DB = {}
RESPONSES_DB = {}

# ----- Resume Parsing -----
def parse_resume(file: UploadFile) -> str:
    ext = file.filename.split('.')[-1].lower()
    if ext == 'pdf':
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif ext == 'docx':
        doc = docx.Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# ----- LangChain-Driven Functions -----
def generate_questions(job_role: str, resume_text: Optional[str] = None) -> List[Question]:
    result = question_chain.run(job_role=job_role, resume_text=resume_text or "")
    questions = result.strip().split("\n")
    categories = ["Technical", "Behavioral", "HR", "Situational"]
    return [
        Question(
            id=str(uuid.uuid4()),
            text=q.strip(),
            category=categories[i % len(categories)]
        ) for i, q in enumerate(questions) if q.strip()
    ]

def evaluate_answer(question_id: str, question: str, answer: str) -> EvaluationResult:
    result = eval_chain.run(question=question, answer=answer)
    lines = result.strip().split("\n")
    confidence = clarity = relevance = 0.0
    feedback = expected = ""
    for line in lines:
        if "Confidence" in line:
            confidence = float(line.split(":")[-1].strip())
        elif "Clarity" in line:
            clarity = float(line.split(":")[-1].strip())
        elif "Relevance" in line:
            relevance = float(line.split(":")[-1].strip())
        elif "Feedback" in line:
            feedback = line.split(":", 1)[-1].strip()
        elif "Expected" in line:
            expected = line.split(":", 1)[-1].strip()
    return EvaluationResult(
        question_id=question_id,
        confidence=confidence,
        clarity=clarity,
        relevance=relevance,
        feedback=feedback,
        expected_answer=expected
    )

def generate_summary_report(user_id: str, evaluations: List[EvaluationResult]) -> EvaluationReport:
    avg_score = sum([(e.confidence + e.clarity + e.relevance)/3 for e in evaluations]) / len(evaluations)
    strengths = [cat for cat in ["Confidence", "Clarity", "Relevance"] if any(getattr(e, cat.lower()) > 8.5 for e in evaluations)]
    weaknesses = [cat for cat in ["Confidence", "Clarity", "Relevance"] if any(getattr(e, cat.lower()) < 6.5 for e in evaluations)]
    plan = "Work on " + ", ".join(weaknesses) if weaknesses else "You're doing great! Keep practicing."
    return EvaluationReport(
        user_id=user_id,
        evaluations=evaluations,
        average_score=avg_score,
        strengths=strengths,
        weaknesses=weaknesses,
        improvement_plan=plan
    )

# ----- API Endpoints -----
@app.post("/start-interview")
def start_interview(job_role: str = Form(...), resume: Optional[UploadFile] = File(None)):
    user_id = str(uuid.uuid4())
    resume_text = parse_resume(resume) if resume else ""
    questions = generate_questions(job_role, resume_text)
    QUESTIONS_DB[user_id] = questions
    return {"user_id": user_id, "questions": questions}

@app.post("/submit-answers")
def submit_answers(user_id: str, answers: List[AnswerInput]):
    evaluations = [evaluate_answer(ans.question_id, ans.question, ans.answer) for ans in answers]
    RESPONSES_DB[user_id] = evaluations
    report = generate_summary_report(user_id, evaluations)
    return report

@app.get("/next-round")
def next_round(user_id: str):
    last_scores = RESPONSES_DB.get(user_id, [])
    if not last_scores:
        return {"error": "No previous session found."}
    avg_score = sum([(e.confidence + e.clarity + e.relevance)/3 for e in last_scores]) / len(last_scores)
    if avg_score >= 9.0:
        return {"message": "You’ve achieved full rating! No more questions needed."}
    # Generate next 10 questions, focus on weak areas (mocked here)
    questions = generate_questions("generic")
    QUESTIONS_DB[user_id] = questions
    return {"questions": questions}
