import re
from datetime import datetime

import pymupdf
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="ResumeIQ | Resume Analyzer",
    page_icon="📄",
    layout="wide",
)


# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
    color: #1e293b;
}

[data-testid="stHeader"] {
    background-color: #f3f6fb;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

h1, h2, h3, h4 {
    color: #172554 !important;
}

.stApp p,
.stApp label,
.stApp [data-testid="stCaptionContainer"],
.stApp .stMarkdown {
    color: #334155;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #173b70, #3975b9);
    color: #ffffff;
    margin-bottom: 24px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 36px;
}

.hero p {
    color: #e6efff !important;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dbe4f0;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #475569 !important;
}

div[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
}

.skill {
    display: inline-block;
    background: #dcfce7;
    color: #166534 !important;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 8px;
}

.missing {
    display: inline-block;
    background: #ffedd5;
    color: #9a3412 !important;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 8px;
}

.section-card {
    background: #ffffff;
    border: 1px solid #dbe4f0;
    border-radius: 14px;
    padding: 18px;
    color: #1e293b;
}

.section-card h3 {
    color: #172554 !important;
}

.stTextArea textarea,
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    color: #1e293b;
}

.stButton button,
.stDownloadButton button {
    border-radius: 9px;
    font-weight: 600;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #eef3fa !important;
}

[data-testid="stExpander"] pre,
[data-testid="stExpander"] pre span,
[data-testid="stExpander"] code,
[data-testid="stCode"] pre,
[data-testid="stCode"] code {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    background-color: #f8fafc !important;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    opacity: 1 !important;
}

.stTextArea label,
.stTextArea label p {
    color: #173b70 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SKILL DATABASE ----------------

SKILL_ALIASES = {
    "python": ["python", "python programming"],
    "sql": ["sql", "structured query language"],
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres"],
    "java": ["java", "core java"],
    "c++": ["c++", "cpp"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "javascript": ["javascript", "js"],
    "react": ["react", "react.js", "reactjs"],
    "fastapi": ["fastapi", "fast api"],
    "flask": ["flask"],
    "django": ["django"],
    "node.js": ["node.js", "nodejs", "node js"],
    "mongodb": ["mongodb", "mongo db"],
    "git": ["git"],
    "github": ["github", "git hub"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "artificial intelligence": [
        "artificial intelligence", "ai"
    ],
    "natural language processing": [
        "natural language processing", "nlp"
    ],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "data analysis": ["data analysis", "data analytics"],
    "excel": ["excel", "microsoft excel"],
    "power bi": ["power bi", "powerbi"],
    "rest api": ["rest api", "restful api", "restful apis"],
    "data structures": ["data structures", "data structure"],
    "oop": [
        "object-oriented programming",
        "object oriented programming",
        "oops", "oop"
    ],
    "communication": ["communication skills", "communication"],
    "problem solving": [
        "problem solving", "problem-solving"
    ],
    "teamwork": ["teamwork", "team work"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "opencv": ["opencv"],
    "docker": ["docker"],
    "linux": ["linux"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "tableau": ["tableau"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "api testing": ["api testing"],
    "pytest": ["pytest"],
    "unit testing": ["unit testing"],
    "agile": ["agile methodology", "agile"],
    "data visualization": ["data visualization"],
}


# ---------------- ANALYSIS FUNCTIONS ----------------

def contains_term(text, term):
    """Match a term without matching it inside unrelated words."""

    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

    return re.search(
        pattern,
        text,
        re.IGNORECASE
    ) is not None


def extract_pdf_text(uploaded_file):
    """Extract selectable text from a PDF."""

    with pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf",
    ) as document:

        return "\n".join(
            page.get_text() for page in document
        ).strip()


def detect_skills(text):
    """Return canonical skill names found in text."""

    found = set()

    for skill, aliases in SKILL_ALIASES.items():

        if any(
            contains_term(text, alias)
            for alias in aliases
        ):
            found.add(skill)

    return found


def text_similarity(resume_text, job_text):
    """Calculate TF-IDF cosine similarity."""

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:

        vectors = vectorizer.fit_transform(
            [resume_text, job_text]
        )

        return round(
            float(
                cosine_similarity(
                    vectors[0:1],
                    vectors[1:2]
                )[0][0]
            ) * 100,
            1,
        )

    except ValueError:
        return 0.0


def check_sections(text):
    """Check whether common resume sections are present."""

    checks = {
        "Contact information": [
            "email", "phone", "mobile", "linkedin"
        ],

        "Education": [
            "education", "academic"
        ],

        "Technical skills": [
            "technical skills", "skills", "technologies"
        ],

        "Projects": [
            "projects", "project experience"
        ],

        "Experience / Internship": [
            "experience", "internship", "intern"
        ],
    }

    result = {}

    for section, keywords in checks.items():

        result[section] = any(
            contains_term(text, keyword)
            for keyword in keywords
        )

    return result


def make_report(
    filename,
    skill_score,
    similarity,
    matched,
    missing,
    sections,
):
    """Generate a downloadable plain-text report."""

    lines = [
        "RESUMEIQ - RESUME ANALYSIS REPORT",
        "=" * 38,

        f"Resume: {filename}",

        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",

        "",

        f"Job skill coverage: {skill_score}%",
        f"TF-IDF text similarity: {similarity}%",

        "",

        "MATCHED SKILLS",
        "-" * 20,

        *[
            f"- {skill}"
            for skill in sorted(matched)
        ],

        "",

        "SKILLS TO REVIEW",
        "-" * 20,

        *[
            f"- {skill}"
            for skill in sorted(missing)
        ],

        "",

        "RESUME SECTION CHECK",
        "-" * 20,

        *[
            f"- {name}: {'Found' if present else 'Not detected'}"
            for name, present in sections.items()
        ],

        "",

        "Note: Scores are text-based estimates, not hiring "
        "predictions. Section checks may miss differently "
        "worded headings. Verify all results manually.",
    ]

    return "\n".join(lines)


# ---------------- MAIN INTERFACE ----------------

st.markdown("""
<div class="hero">
    <h1>📄 ResumeIQ</h1>
    <p>Resume analysis, skill matching, and job-readiness insights.</p>
</div>
""", unsafe_allow_html=True)


st.write(
    "Compare a resume with a job description to identify "
    "relevant skills and areas to review."
)


st.info(
    "Privacy: this version processes the uploaded resume "
    "during the current app session and does not save it "
    "to a database."
)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("About ResumeIQ")

    st.write("**Version:** 2.0")

    st.write("**Engine:** Python + NLP features")

    st.write(
        "**Analysis:** Skill coverage and TF-IDF similarity"
    )

    st.divider()

    st.caption(
        "Only add skills to your resume if you genuinely "
        "have those skills."
    )


# ---------------- RESUME UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"],
)


# ---------------- JOB DESCRIPTION ----------------

job_description = st.text_area(
    "Paste the target job description",
    height=220,
    placeholder=(
        "Paste the responsibilities, required skills, "
        "qualifications, and technologies here..."
    ),
)


# ---------------- ANALYZE BUTTON ----------------

analyze = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
)


# ---------------- ANALYSIS PROCESS ----------------

if analyze:

    if uploaded_file is None:

        st.error("Please upload a PDF resume.")

    elif not job_description.strip():

        st.error("Please enter a job description.")

    else:

        try:

            with st.spinner("Analyzing your resume..."):

                # Extract PDF text
                resume_text = extract_pdf_text(uploaded_file)

                if len(resume_text) < 30:

                    st.error(
                        "Very little readable text was found. "
                        "This PDF may be scanned. Try a "
                        "text-based PDF resume."
                    )

                    st.stop()

                # Detect resume skills
                resume_skills = detect_skills(resume_text)

                # Detect job description skills
                job_skills = detect_skills(job_description)

                # Compare skills
                matched = resume_skills & job_skills

                missing = job_skills - resume_skills

                # Calculate skill coverage
                skill_score = (
                    round(
                        len(matched) / len(job_skills) * 100,
                        1
                    )
                    if job_skills else 0.0
                )

                # Calculate text similarity
                similarity = text_similarity(
                    resume_text,
                    job_description
                )

                # Check resume sections
                sections = check_sections(resume_text)

            # ---------------- SUCCESS ----------------

            st.success("Analysis completed!")

            st.subheader("Analysis Dashboard")

            # ---------------- METRICS ----------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Job Skill Coverage",
                f"{skill_score}%"
            )

            col2.metric(
                "Text Similarity",
                f"{similarity}%"
            )

            col3.metric(
                "Matched Skills",
                len(matched)
            )

            col4.metric(
                "Skills to Review",
                len(missing)
            )

            st.caption(
                "Skill coverage is the proportion of detected job "
                "description skills found in the resume. Text "
                "similarity measures document similarity, not "
                "candidate quality or hiring probability."
            )

            # ---------------- SKILL RESULTS ----------------

            left, right = st.columns(2)

            # Matched skills
            with left:

                st.markdown(
                    '<div class="section-card">'
                    '<h3>✅ Matched Skills</h3>',
                    unsafe_allow_html=True,
                )

                if matched:

                    st.markdown(
                        "".join(
                            f'<span class="skill">{skill}</span>'
                            for skill in sorted(matched)
                        ),
                        unsafe_allow_html=True,
                    )

                else:

                    st.write(
                        "No supported matching skills detected."
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            # Missing skills
            with right:

                st.markdown(
                    '<div class="section-card">'
                    '<h3>📌 Skills to Review</h3>',
                    unsafe_allow_html=True,
                )

                if missing:

                    st.markdown(
                        "".join(
                            f'<span class="missing">{skill}</span>'
                            for skill in sorted(missing)
                        ),
                        unsafe_allow_html=True,
                    )

                else:

                    st.write(
                        "No missing skills detected from "
                        "the supported skill list."
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            # ---------------- SECTION REVIEW ----------------

            st.subheader("Resume Section Review")

            section_cols = st.columns(len(sections))

            for column, (name, present) in zip(
                section_cols,
                sections.items()
            ):

                with column:

                    st.write(
                        "✅" if present else "⚠️"
                    )

                    st.caption(name)

                    st.write(
                        "Found" if present else "Not detected"
                    )

            # ---------------- IMPROVEMENT SUGGESTIONS ----------------

            st.subheader("Improvement Suggestions")

            if missing:

                st.write(
                    "Review these job-related skills and identify "
                    "which ones you can genuinely demonstrate."
                )

                for skill in sorted(missing):

                    st.write(
                        f"- Review your knowledge of **{skill}**."
                    )

            if not sections.get("Projects", False):

                st.write(
                    "- Consider including a clearly labelled "
                    "Projects section if you have relevant projects."
                )

            if not sections.get("Education", False):

                st.write(
                    "- Check that your education details are "
                    "easy to find."
                )

            if skill_score < 50 and job_skills:

                st.write(
                    "- Compare the job requirements with your "
                    "actual skills and project experience."
                )

            # ---------------- DOWNLOAD REPORT ----------------

            st.subheader("Download Your Report")

            report = make_report(
                uploaded_file.name,
                skill_score,
                similarity,
                matched,
                missing,
                sections,
            )

            st.download_button(
                "⬇️ Download Analysis Report",
                data=report,
                file_name="resume_analysis_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # ---------------- EXTRACTED RESUME TEXT ----------------

            with st.expander("View extracted resume text"):

                st.text_area(
                    "Extracted Resume Content",
                    value=resume_text,
                    height=400,
                    disabled=False,
                    key="resume_text_display",
                )

        except Exception as error:

            st.error(
                "Unable to analyze this PDF. Check that the "
                "file is valid and try again."
            )

            st.caption(
                f"Technical details: {error}"
            )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "ResumeIQ | Educational tool. Review results manually "
    "before making career decisions."
)