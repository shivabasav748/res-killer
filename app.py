# app.py
import os
import io
import tempfile
import base64
from typing import List
import pandas as pd
import streamlit as st

# parsing + NLP
import fitz  # pymupdf
import docx2txt
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# optional OpenAI import handled safely
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# -------- Page config & caching --------
st.set_page_config(page_title="res killer " \
"A Resume Relevance Checker", page_icon="", layout="wide")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# -------- CSS for styling --------
st.markdown(
    """
    <style>
    .app-header {display:flex; align-items:center; gap:12px;}
    .app-title {font-size:28px; font-weight:700;}
    .card {
        background: linear-gradient(180deg, #ffffff, #f7f9fc);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(24,39,75,0.08);
        margin-bottom: 12px;
    }
    .metric-title {font-size:14px; color:#6b7280}
    .metric-value {font-size:20px; font-weight:700}
    .stButton>button {border-radius:8px; padding:10px 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------- Helpers: text extraction --------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    text = docx2txt.process(tmp_path)
    return text

def extract_text_from_uploaded(uploaded_file) -> str:
    data = uploaded_file.read()
    fname = uploaded_file.name.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    elif fname.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    else:
        # fallback: try decode
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

# -------- Skill extraction & matching --------
COMMON_SKILLS = [
    "python","sql","machine learning","deep learning","django","flask",
    "aws","docker","kubernetes","pandas","numpy","pytorch","tensorflow",
    "nlp","java","c++","excel","communication","leadership","react","node"
]

def extract_skills_from_jd(jd_text: str) -> List[str]:
    skills = []
    low = jd_text.lower()
    for s in COMMON_SKILLS:
        if s in low or fuzz.partial_ratio(s, low) > 85:
            skills.append(s)
    return skills

def hard_match_score(resume_text: str, jd_skills: List[str], threshold: int = 70):
    resume_lower = resume_text.lower()
    matched = []
    for skill in jd_skills:
        score = fuzz.partial_ratio(skill.lower(), resume_lower)
        if score >= threshold:
            matched.append(skill)
    score_percent = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0
    return score_percent, matched

def semantic_score(resume_text: str, jd_text: str) -> float:
    # encode and compute cosine similarity
    emb_r = model.encode(resume_text, convert_to_tensor=True)
    emb_j = model.encode(jd_text, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_r, emb_j).item()  # [-1,1]
    if sim < 0: sim = 0.0
    return sim * 100.0

def combined_score(hard: float, semantic: float, hard_weight: float, semantic_weight: float) -> float:
    return hard_weight * hard + semantic_weight * semantic

def verdict_from_score(score: float) -> str:
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

# -------- Optional: LLM feedback (OpenAI) --------
def generate_llm_feedback(jd_text: str, resume_text: str, missing_skills: List[str]) -> str:
    if not HAS_OPENAI:
        return "LLM feedback unavailable (openai package not installed)."
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "LLM feedback unavailable (OPENAI_API_KEY not set)."
    try:
        openai.api_key = key
        prompt = f"""
You are a hiring coach. Job Description:
{jd_text}

Candidate Resume (short excerpt):
{resume_text[:2000]}

Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}

Provide:
1) 3 concise bullet suggestions to improve the resume for this job.
2) One suggested bullet point (single line) the candidate can add to their resume to demonstrate fit.
Return plain text.
"""
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM feedback failed: {e}"

# -------- UI: header & sidebar --------
with st.container():
    col1, col2 = st.columns([1,10])
    with col1:
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=72)
    with col2:
        st.markdown('<div class="app-header"><div class="app-title">📄 Resume Relevance Checker</div></div>', unsafe_allow_html=True)
        st.markdown("A fast MVP to evaluate resumes vs job descriptions. Upload a JD and candidate resumes to get a relevance score, missing skills, and optional improvement tips.")

st.sidebar.title("Settings & Navigation")
page = st.sidebar.radio("Page", ["Evaluate", "Results", "About"])
# scoring weights
st.sidebar.markdown("### Scoring settings")
hard_weight = st.sidebar.slider("Hard-match weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
semantic_weight = st.sidebar.slider("Semantic weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
threshold = st.sidebar.slider("Skill fuzzy threshold", min_value=50, max_value=95, value=70, step=1)

# session storage for results
if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame()

# -------- Page: Evaluate --------
if page == "Evaluate":
    st.markdown("## Upload Job Description")
    jd_input = st.text_area("Paste Job Description text (or upload a JD file below)", height=180)
    jd_file = st.file_uploader("Optional: Upload JD file (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    if jd_file is not None and not jd_input:
        jd_input = extract_text_from_uploaded(jd_file)

    st.markdown("## Upload candidate resumes (multiple)")
    uploads = st.file_uploader("Upload resumes (PDF / DOCX)", type=["pdf","docx"], accept_multiple_files=True)

    col_eval_left, col_eval_right = st.columns([3,1])
    with col_eval_left:
        evaluate_btn = st.button("Evaluate Resumes", type="primary")
    with col_eval_right:
        st.markdown("## Quick tips")
        st.write("- Paste full JD for best results\n- Upload cleaned resumes (not scanned images)")

    if evaluate_btn:
        if not jd_input or not uploads:
            st.error("Please provide a Job Description and at least one resume.")
        else:
            jd_skills = extract_skills_from_jd(jd_input)
            st.markdown("**Detected skills from JD:** " + (", ".join(jd_skills) if jd_skills else "None detected (consider pasting more details)."))
            results = []
            progress = st.progress(0)
            n = len(uploads)
            for i, f in enumerate(uploads):
                try:
                    resume_text = extract_text_from_uploaded(f)
                except Exception as e:
                    resume_text = ""
                hard, matched = hard_match_score(resume_text, jd_skills, threshold=threshold)
                semantic = semantic_score(resume_text, jd_input)
                final = combined_score(hard, semantic, hard_weight, semantic_weight)
                verdict = verdict_from_score(final)
                missing = [s for s in jd_skills if s not in matched]
                feedback = generate_llm_feedback(jd_input, resume_text, missing) if HAS_OPENAI and os.getenv("OPENAI_API_KEY") else "LLM feedback unavailable (set OPENAI_API_KEY to enable)"
                results.append({
                    "resume": f.name,
                    "score": round(final,2),
                    "verdict": verdict,
                    "matched": ", ".join(matched),
                    "missing": ", ".join(missing) if missing else "None",
                    "feedback": feedback,
                })
                progress.progress(int(((i+1)/n)*100))
            df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
            st.session_state["results_df"] = df
            st.success("Evaluation complete! See results on the Results page or below.")
            st.dataframe(df)

# -------- Page: Results --------
if page == "Results":
    df = st.session_state.get("results_df", pd.DataFrame())
    if df.empty:
        st.info("No results yet. Run an evaluation on the Evaluate page first.")
    else:
        avg_score = df["score"].mean()
        high = (df["verdict"] == "High").sum()
        medium = (df["verdict"] == "Medium").sum()
        low = (df["verdict"] == "Low").sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average Score", f"{avg_score:.1f}")
        c2.metric("High fit", int(high))
        c3.metric("Medium fit", int(medium))
        c4.metric("Low fit", int(low))

        st.markdown("### Scores")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", csv, file_name="resume_results.csv", mime="text/csv")

        st.markdown("### Score distribution")
        st.bar_chart(df.set_index("resume")["score"])

        st.markdown("### Candidate details")
        for i, row in df.iterrows():
            with st.expander(f"{row['resume']} — Score {row['score']} ({row['verdict']})"):
                st.markdown(f"**Matched skills:** {row['matched']}")
                st.markdown(f"**Missing skills:** {row['missing']}")
                st.markdown("**LLM Feedback:**")
                st.write(row["feedback"])

# -------- Page: About --------
if page == "About":
    st.markdown("## About this MVP")
    st.markdown("""
    - Uses rule-based skill detection + embedding (semantic) matching.
    - Sentence-transformers (all-MiniLM-L6-v2) for semantic similarity.
    - RapidFuzz for fuzzy skill matching.
    - Optional OpenAI integration for personalized feedback (set OPENAI_API_KEY).
    """)
    st.markdown("## Next improvements (ideas)")
    st.write("""
    - Add resume parsing & sections normalization (skills/experience/education).
    - Cache embeddings & use a vector DB (Chroma) for scalability.
    - OCR for scanned resumes (Tesseract).
    - Add user accounts + database to store history.
    """)
