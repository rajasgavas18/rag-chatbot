import streamlit as st
from groq import Groq
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API Key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Page Setup
st.set_page_config(page_title="Smart Document Chatbot", page_icon="🤖", layout="wide")

# Custom CSS — Better UI
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTitle { color: #00d4ff !important; font-size: 2.5rem !important; }
    .score-box {
        background: linear-gradient(135deg, #1e3a5f, #0e1117);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .feature-badge {
        background: #00d4ff;
        color: black;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 Smart Document Chatbot")
st.markdown("**PDF upload करा → प्रश्न विचारा → Instant AI Answers!**")
st.divider()

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "all_text" not in st.session_state:
    st.session_state.all_text = ""
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    language = st.selectbox("🌐 Language / भाषा", 
                           ["English", "मराठी", "हिंदी"])
    
    st.divider()
    st.markdown("### 📁 Uploaded PDFs")
    if st.session_state.pdf_names:
        for name in st.session_state.pdf_names:
            st.success(f"✅ {name}")
    else:
        st.info("कोणताही PDF नाही")
    
    st.divider()
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.all_text = ""
        st.session_state.pdf_names = []
        st.rerun()

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📄 PDF Upload करा")
    uploaded_files = st.file_uploader(
        "एक किंवा अनेक PDF select करा",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.pdf_names]
        if new_files:
            with st.spinner("📖 PDFs वाचत आहे..."):
                for uploaded_file in new_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(pages)
                    st.session_state.all_text += "\n".join([c.page_content for c in chunks])
                    st.session_state.pdf_names.append(uploaded_file.name)
                    os.unlink(tmp_path)
            st.success(f"✅ {len(new_files)} PDF ready!")

with col2:
    # Resume Score Feature
    if st.session_state.all_text and st.button("📊 Resume Score द्या!", use_container_width=True, type="primary"):
        with st.spinner("Resume analyze करत आहे..."):
            if language == "मराठी":
                score_prompt = f"""हा resume आहे. मराठीत उत्तर दे.
खालील गोष्टी analyze कर आणि score दे (100 पैकी):
1. Skills (25 पैकी)
2. Education (25 पैकी)  
3. Experience/Projects (25 पैकी)
4. Overall Presentation (25 पैकी)

Resume: {st.session_state.all_text[:3000]}

Format: 
🎯 Total Score: X/100
📚 Skills: X/25
🎓 Education: X/25  
💼 Experience: X/25
✨ Presentation: X/25
💡 Top 3 Improvements:"""
            elif language == "हिंदी":
                score_prompt = f"""यह resume है। हिंदी में जवाब दो।
Resume analyze करो और score दो (100 में से):
Resume: {st.session_state.all_text[:3000]}

Format:
🎯 Total Score: X/100
📚 Skills: X/25
🎓 Education: X/25
💼 Experience: X/25
✨ Presentation: X/25
💡 Top 3 Improvements:"""
            else:
                score_prompt = f"""Analyze this resume and give a score out of 100.
Resume: {st.session_state.all_text[:3000]}

Format:
🎯 Total Score: X/100
📚 Skills: X/25
🎓 Education: X/25
💼 Experience: X/25
✨ Presentation: X/25
💡 Top 3 Improvements:"""

            score_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": score_prompt}]
            )
            score_text = score_response.choices[0].message.content
            st.markdown('<div class="score-box">', unsafe_allow_html=True)
            st.markdown(score_text)
            st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Chat Section
st.markdown("### 💬 Chat करा")

if st.session_state.all_text:
    # Language hint
    if language == "मराठी":
        placeholder = "मराठीत प्रश्न विचारा... उदा: माझ्या resume मध्ये काय skills आहेत?"
    elif language == "हिंदी":
        placeholder = "हिंदी में सवाल पूछें... जैसे: मेरे resume में क्या skills हैं?"
    else:
        placeholder = "Ask in English... e.g: What are my top skills?"

    user_question = st.chat_input(placeholder)

    if user_question:
        # Build chat history for memory
        history_text = ""
        for chat in st.session_state.chat_history[-4:]:
            history_text += f"User: {chat['user']}\nAssistant: {chat['bot']}\n\n"

        if language == "मराठी":
            lang_instruction = "मराठीत उत्तर दे."
        elif language == "हिंदी":
            lang_instruction = "हिंदी में जवाब दो।"
        else:
            lang_instruction = "Answer in English."

        prompt = f"""तू एक helpful assistant आहेस. {lang_instruction}

मागील conversation:
{history_text}

Document:
{st.session_state.all_text[:3000]}

नवीन प्रश्न: {user_question}

उत्तर नीट आणि clear दे."""

        with st.spinner("AI विचार करत आहे..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content

        st.session_state.chat_history.append({
            "user": user_question,
            "bot": answer
        })

    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["bot"])

else:
    st.info("👆 आधी PDF upload करा — मग chat करता येईल!")