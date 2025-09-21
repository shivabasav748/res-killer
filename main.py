# main.py

import fitz  # PyMuPDF library
from fastapi import FastAPI, UploadFile, File, Form
from typing import Annotated

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Automated Resume Relevance Checker",
    description="An AI-powered system to score resumes against job descriptions."
)


# 2. Create the function to extract text from a PDF
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from a PDF file's content."""
    text = ""
    try:
        # Open the PDF from the in-memory file content
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            # Iterate through each page and extract text
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return f"Error processing PDF: {e}"
    return text


def score_resume_against_jd(resume_text: str, job_description: str) -> float:
    """Simple scoring: percentage of JD keywords found in resume."""
    jd_keywords = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())
    matches = jd_keywords & resume_words
    if not jd_keywords:
        return 0.0
    return round(len(matches) / len(jd_keywords) * 100, 2)


# 3. Define the API endpoint for analysis
@app.post("/analyze/")
async def analyze_resume(
    resume_file: Annotated[UploadFile, File(description="The candidate's resume in PDF format.")],
    job_description: Annotated[str, Form(description="The job description text.")]
):
    """
    This endpoint accepts a resume (PDF) and a job description (text),
    and returns the extracted text from the resume.
    --- This is our MVP for Phase 1 ---
    """
    # Read the content of the uploaded resume file
    resume_content = await resume_file.read()
    
    # Use our function to extract the text
    extracted_text = extract_text_from_pdf(resume_content)
    score = score_resume_against_jd(extracted_text, job_description)
    return {
        "resume_filename": resume_file.filename,
        "job_description_received": job_description,
        "extracted_resume_text": extracted_text,
        "relevance_score": score
    }

# A simple root endpoint to confirm the server is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Analyzer API!"}