# app.py
import os
import tempfile
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

# Parsing and NLP
import fitz  # PyMuPDF
import docx2txt
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# Optional OpenAI
try:
    import openai
    HAS_OPENAI = True
except:
    HAS_OPENAI = False

# Page config
st.set_page_config(page_title="Res Killer", page_icon="", layout="wide")

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

# Inject CSS
st.markdown("""
<style>
body {background-color: #f4f7f9;}
.stButton>button {
    background-color: #2ecc71; color: white; font-size: 18px; border-radius: 10px; padding: 10px 20px;
}
.stFileUploader {border: 2px dashed #3498db; border-radius: 12px; padding: 15px; background: #ecf6ff;}
.card {background: #ffffff; border-radius:12px; padding:16px; box-shadow:0 6px 18px rgba(24,39,75,0.08); margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

# Helper functions
def extract_text_from_pdf_bytes(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join([page.get_text() for page in doc])

def extract_text_from_docx_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return docx2txt.process(tmp_path)

def extract_text(uploaded_file):
    data = uploaded_file.read()
    fname = uploaded_file.name.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    elif fname.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    else:
        return data.decode("utf-8", errors="ignore")

COMMON_SKILLS = ["python","sql","machine learning","deep learning","django","flask","aws","docker","kubernetes","pandas","numpy","pytorch","tensorflow","nlp","java","c++","excel","communication","leadership","react","node"]

def extract_skills(text):
    skills = []
    low = text.lower()
    for s in COMMON_SKILLS:
        if s in low or fuzz.partial_ratio(s, low) > 85:
            skills.append(s)
    return skills

def hard_match(resume_text, jd_skills, threshold=70):
    matched = []
    for skill in jd_skills:
        score = fuzz.partial_ratio(skill.lower(), resume_text.lower())
        if score >= threshold:
            matched.append(skill)
    return (len(matched)/len(jd_skills))*100 if jd_skills else 0, matched

def semantic_score(resume_text, jd_text):
    emb_r = model.encode(resume_text, convert_to_tensor=True)
    emb_j = model.encode(jd_text, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_r, emb_j).item()
    return max(sim,0)*100

def combined_score(hard, semantic, hard_weight, semantic_weight):
    return hard_weight*hard + semantic_weight*semantic

def verdict(score):
    if score>=75: return "High"
    elif score>=50: return "Medium"
    else: return "Low"

def generate_feedback(jd_text, resume_text, missing_skills):
    if not HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return "LLM feedback unavailable"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"JD:\n{jd_text}\nResume snippet:\n{resume_text[:2000]}\nMissing skills: {', '.join(missing_skills)}\nProvide 3 concise suggestions + 1 bullet to add to resume."
    resp = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
        messages=[{"role":"user","content":prompt}],
        max_tokens=300, temperature=0.2
    )
    return resp["choices"][0]["message"]["content"].strip()

# Sidebar
st.sidebar.title("⚙️ Settings")
page = st.sidebar.radio("Navigation", ["Evaluate","Results","About"])
hard_weight = st.sidebar.slider("Hard Match Weight",0.0,1.0,0.4,0.05)
semantic_weight = st.sidebar.slider("Semantic Weight",0.0,1.0,0.6,0.05)
threshold = st.sidebar.slider("Skill Threshold",50,95,70,1)

if "results_df" not in st.session_state: st.session_state["results_df"] = pd.DataFrame()

# Header
st.title("⚡ Res Killer – AI Resume Evaluator")
st.subheader("Smart Resume Relevance Scoring System")

# Page: Evaluate
if page=="Evaluate":
    st.markdown("### Upload Job Description")
    jd_text = st.text_area("Paste JD here", height=180)
    jd_file = st.file_uploader("Or upload JD file", type=["pdf","docx","txt"])
    if jd_file and not jd_text: jd_text = extract_text(jd_file)

    st.markdown("### Upload Candidate Resumes")
    uploads = st.file_uploader("Upload resumes (multiple)", type=["pdf","docx"], accept_multiple_files=True)

    if st.button("Evaluate Resumes"):
        if not jd_text or not uploads:
            st.error("Provide JD and at least 1 resume")
        else:
            jd_skills = extract_skills(jd_text)
            st.markdown("**Detected JD skills:** "+(", ".join(jd_skills) if jd_skills else "None"))
            results = []
            progress = st.progress(0)
            n = len(uploads)
            for i,f in enumerate(uploads):
                r_text = extract_text(f)
                hard, matched = hard_match(r_text,jd_skills,threshold)
                sem = semantic_score(r_text,jd_text)
                final = combined_score(hard,sem,hard_weight,semantic_weight)
                verdict_text = verdict(final)
                missing = [s for s in jd_skills if s not in matched]
                feedback = generate_feedback(jd_text,r_text,missing)
                results.append({
                    "resume": f.name,
                    "score": round(final,2),
                    "verdict": verdict_text,
                    "matched": ", ".join(matched),
                    "missing": ", ".join(missing) if missing else "None",
                    "feedback": feedback
                })
                progress.progress(int((i+1)/n*100))
            df = pd.DataFrame(results).sort_values("score",ascending=False)
            st.session_state["results_df"] = df
            st.success("Evaluation complete!")
            st.dataframe(df)

# Page: Results
if page=="Results":
    df = st.session_state.get("results_df", pd.DataFrame())
    if df.empty: st.info("No results yet. Run Evaluate first")
    else:
        avg_score = df["score"].mean()
        high = (df["verdict"]=="High").sum()
        medium = (df["verdict"]=="Medium").sum()
        low = (df["verdict"]=="Low").sum()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Average Score",f"{avg_score:.1f}")
        c2.metric("High Fit",int(high))
        c3.metric("Medium Fit",int(medium))
        c4.metric("Low Fit",int(low))

        st.markdown("### Detailed Results")
        st.dataframe(df,use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV",csv,"resume_results.csv","text/csv")

        st.markdown("### Candidate Details")
        for _,row in df.iterrows():
            with st.expander(f"{row['resume']} — Score {row['score']} ({row['verdict']})"):
                st.markdown(f"**Matched Skills:** {row['matched']}")
                st.markdown(f"**Missing Skills:** {row['missing']}")
                st.markdown("**Feedback:**")
                st.write(row["feedback"])

                # Skill match chart
                matched_skills = row['matched'].split(", ") if row['matched'] != "None" else []
                missing_skills = row['missing'].split(", ") if row['missing'] != "None" else []
                all_skills = matched_skills + missing_skills
                values = [1]*len(matched_skills) + [0]*len(missing_skills)
                colors = ["#2ecc71"]*len(matched_skills) + ["#e74c3c"]*len(missing_skills)
                if all_skills:
                    fig = go.Figure(go.Bar(
                        x=values,
                        y=all_skills,
                        orientation='h',
                        marker_color=colors,
                        text=["✅" if v==1 else "❌" for v in values],
                        textposition="outside"
                    ))
                    fig.update_layout(
                        title="Skill Match Overview",
                        xaxis=dict(title="Matched = 1 / Missing = 0", showticklabels=False),
                        yaxis=dict(autorange="reversed"),
                        height=30*len(all_skills)+100,
                        margin=dict(l=100, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # JS confetti for High fit
                if row['verdict']=="High":
                    confetti = """
                    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.4.0/dist/confetti.browser.min.js"></script>
                    <script>
                    confetti({particleCount: 100, spread: 70, origin:{y:0.6}});
                    </script>
                    """
                    components.html(confetti, height=100)

# Page: About
if page=="About":
    st.markdown("## About Res Killer")
    st.write("""
    AI-powered Resume Relevance Checker.
    - Hard + Semantic match scoring
    - Optional LLM feedback
    - Streamlit Dashboard with progress and metrics
    """)
    st.markdown("## Next Features Ideas")
    st.write("""
    - Resume section parsing (Skills, Experience, Education)
    - Vector DB for faster embedding search
    - OCR for scanned resumes
    - User login + database history
    """)
