import streamlit as st
from google.generativeai import GenerativeModel, configure
from PyPDF2 import PdfReader
from docx import Document
import json

# Secure Gemini API key
try:
    configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Gemini API key not found. Add GEMINI_API_KEY to Streamlit secrets.")
    st.stop()

# Updated model name to 'gemini-2.5-flash' (valid as of January 2026)
model = GenerativeModel('gemini-2.5-flash')

st.set_page_config(page_title="EchoMind", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    .big-font { font-size:50px !important; font-weight:bold; text-align:center; color:#00ffaa; }
    .subheader { font-size:24px; color:#cccccc; text-align:center; margin-bottom:40px; }
    .upload-box { border: 2px dashed #00ffaa; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🧠 EchoMind</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Understand why your past self thought, felt, and wrote the way you did.</p>', unsafe_allow_html=True)

st.markdown("### Upload your old content")
st.markdown("<div class='upload-box'>Supported: Text files, PDFs, Word docs, Twitter JSON exports</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop files here or click to browse",
    accept_multiple_files=True,
    type=['txt', 'pdf', 'docx', 'json'],
    label_visibility="collapsed"
)

if uploaded_files:
    all_text = ""
    file_info = []

    for file in uploaded_files:
        content = ""
        filename = file.name

        if file.type == "text/plain":
            content = file.read().decode("utf-8", errors="ignore")
        elif file.type == "application/pdf":
            try:
                reader = PdfReader(file)
                content = "\n".join([page.extract_text() or "" for page in reader.pages])
            except:
                content = "[Error reading PDF]"
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = Document(file)
                content = "\n".join([para.text for para in doc.paragraphs])
            except:
                content = "[Error reading DOCX]"
        elif file.type == "application/json":
            try:
                data = json.load(file)
                if isinstance(data, list):
                    content = "\n".join([item.get('full_text') or item.get('text', '') for item in data])
                else:
                    content = json.dumps(data, indent=2)
            except:
                content = "[Error reading JSON]"

        if content.strip():
            all_text += f"\n\n--- From {filename} ---\n{content[:15000]}"
            file_info.append(filename)

    if all_text.strip():
        with st.spinner("EchoMind is reflecting on your past self..."):
            try:
                response = model.generate_content(f"""
You are EchoMind — an empathetic, insightful analyst who helps people understand their past mindset.

Analyze this content I created in the past (from files: {', '.join(file_info)}):

{all_text[:20000]}

Explain WHY I likely thought, felt, or wrote this way, considering:
- Probable age and life stage
- Language patterns, emotional tone, maturity level
- Cultural, social, or technological context of the time
- Common psychological development

Be kind, non-judgmental, and deeply understanding. Use "you" to speak directly.
Clearly label inferences (e.g., "It seems you were...").
Structure: Insightful paragraphs + bullet points for key factors.
""")
                insight = response.text

                st.success("Analysis complete")
                st.markdown("### Why Your Past Self Thought This Way")
                st.markdown(insight)

                st.caption("EchoMind uses Gemini AI for interpretation — this is an educated reflection, not absolute truth.")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    else:
        st.warning("No readable text found in uploaded files.")

st.markdown("---")
st.caption("EchoMind • Understand your past to know yourself better • Powered by Gemini AI")
