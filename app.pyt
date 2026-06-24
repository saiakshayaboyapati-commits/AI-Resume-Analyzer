import streamlit as st
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
from io import BytesIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🤖 AI Resume Analyzer")

st.sidebar.info("""
Features:
✅ ATS Score
✅ Resume-JD Similarity
✅ Skill Gap Analysis
✅ AI Feedback
✅ Experience Indicator
✅ PDF Report Download
""")

# =========================
# TITLE
# =========================
st.title("🤖 AI Resume Analyzer")
st.markdown(
    "Analyze resumes using NLP, ATS scoring, skill matching, and AI-powered feedback."
)

# =========================
# SKILLS DATABASE
# =========================
skills_db = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "Power BI",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Data Analysis",
    "AWS",
    "Azure",
    "TensorFlow",
    "Tableau",
    "Excel",
    "Git",
    "Data Science",
    "Pandas",
    "NumPy",
    "Docker",
    "Kubernetes"
]

# =========================
# INPUTS
# =========================
uploaded_file = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "📝 Paste Job Description Here"
)

# =========================
# MAIN LOGIC
# =========================
if uploaded_file and job_description:

    # Read PDF
    pdf = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf.pages:
        text = page.extract_text()
        if text:
            resume_text += text

    resume_text_lower = resume_text.lower()
    jd_lower = job_description.lower()

    # =========================
    # SKILL EXTRACTION
    # =========================
    resume_skills = []
    jd_skills = []

    for skill in skills_db:

        if skill.lower() in resume_text_lower:
            resume_skills.append(skill)

        if skill.lower() in jd_lower:
            jd_skills.append(skill)

    matching_skills = list(set(resume_skills) & set(jd_skills))
    missing_skills = list(set(jd_skills) - set(resume_skills))

    # =========================
    # ATS SCORE
    # =========================
    if len(jd_skills) > 0:
        score = (len(matching_skills) / len(jd_skills)) * 100
    else:
        score = 0

    st.success("✅ Analysis Complete!")

    st.subheader("📊 ATS Resume Score")
    st.write(f"{score:.2f}%")
    st.progress(int(score))

    # Resume Strength
    if score >= 80:
        st.success("🏆 Strong Resume")
    elif score >= 60:
        st.warning("⭐ Average Resume")
    else:
        st.error("⚠️ Needs Improvement")

    # =========================
    # NLP SIMILARITY
    # =========================
    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0] * 100

    st.subheader("🧠 Resume-JD Similarity Score")
    st.write(f"{similarity:.2f}%")

    # =========================
    # PIE CHART
    # =========================
    matched = len(matching_skills)
    missing = len(missing_skills)

    if matched > 0 or missing > 0:

        fig, ax = plt.subplots()

        ax.pie(
            [matched, missing],
            labels=["Matched Skills", "Missing Skills"],
            autopct="%1.1f%%"
        )

        st.subheader("📈 Skills Analysis")
        st.pyplot(fig)

    # =========================
    # EXPERIENCE INDICATOR
    # =========================
    project_keywords = [
        "project",
        "projects",
        "internship",
        "experience"
    ]

    project_count = sum(
        resume_text_lower.count(word)
        for word in project_keywords
    )

    st.subheader("📂 Experience Indicator")
    st.write(
        f"Mentions of Projects / Experience: {project_count}"
    )

    # =========================
    # AI FEEDBACK
    # =========================
    st.subheader("🤖 AI Feedback")

    if score >= 80:
        st.success(
            "Excellent Resume Match! Your resume aligns very well with the job description."
        )
    elif score >= 60:
        st.warning(
            "Good Match. Consider adding the missing skills to improve your chances."
        )
    else:
        st.error(
            "Low Match Score. Add more relevant skills, certifications, and projects."
        )

    # =========================
    # SKILLS
    # =========================
    st.subheader("💡 Skills Found in Resume")
    st.write(resume_skills)

    st.subheader("✅ Matching Skills")
    st.write(matching_skills)

    st.subheader("❌ Missing Skills")
    st.write(missing_skills)

    # =========================
    # SUGGESTIONS
    # =========================
    st.subheader("📌 Resume Suggestions")

    if missing_skills:
        for skill in missing_skills:
            st.write(
                f"✅ Consider adding {skill} if you have experience with it."
            )
    else:
        st.write(
            "🎉 Great! No important skills are missing."
        )

    # =========================
    # RESUME TEXT
    # =========================
    st.subheader("📄 Extracted Resume Text")

    st.text_area(
        "Resume Content",
        resume_text,
        height=250
    )

    # =========================
    # PDF REPORT
    # =========================
    pdf_buffer = BytesIO()

    c = canvas.Canvas(pdf_buffer)

    c.drawString(100, 800, "AI Resume Analyzer Report")
    c.drawString(100, 770, f"ATS Score: {score:.2f}%")
    c.drawString(100, 740, f"Similarity Score: {similarity:.2f}%")
    c.drawString(100, 710, f"Experience Indicator: {project_count}")

    c.drawString(100, 680, "Matching Skills:")
    c.drawString(100, 660, ", ".join(matching_skills))

    c.drawString(100, 630, "Missing Skills:")
    c.drawString(100, 610, ", ".join(missing_skills))

    c.save()

    pdf_buffer.seek(0)

    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name="resume_analysis_report.pdf",
        mime="application/pdf"
    )