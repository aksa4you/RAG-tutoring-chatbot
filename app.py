import streamlit as st
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pypdf import PdfReader

st.set_page_config(page_title="RAG Tutoring Chatbot", page_icon="📚")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background with subtle gradient */
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #12151C 100%);
}

/* Title styling */
h1 {
    background: linear-gradient(90deg, #00D9FF, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #14171F;
    border-right: 1px solid #262B36;
}

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    background-color: #1A1D24;
    border: 1px solid #262B36;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

/* File uploader box */
[data-testid="stFileUploader"] {
    background-color: #1A1D24;
    border: 1px dashed #00D9FF44;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00D9FF, #7C3AED);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: transform 0.15s ease;
}
.stButton > button:hover {
    transform: scale(1.03);
}

/* Chat input box */
[data-testid="stChatInput"] {
    border: 1px solid #00D9FF55 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 RAG-Based Tutoring Chatbot")
st.caption("Upload your study material (PDF or text), then ask questions about it.")

# ---------- Configure Gemini ----------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
if not GOOGLE_API_KEY:
    st.error("Gemini API key not found. Add it in Settings → Secrets as GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-flash-latest")


class GeminiEmbeddings(Embeddings):
    """Uses Gemini's embedding API instead of a local model - much lighter
    and faster on limited-memory free hosting."""

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document",
            )
            vectors.append(result["embedding"])
        return vectors

    def embed_query(self, text):
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return GeminiEmbeddings()

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar: build knowledge base ----------
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF or text files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Build Knowledge Base"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Reading and indexing documents..."):
                raw_text = ""
                for file in uploaded_files:
                    if file.name.lower().endswith(".pdf"):
                        reader = PdfReader(file)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                raw_text += text + "\n"
                    else:
                        raw_text += file.read().decode("utf-8") + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=100
                )
                chunks = splitter.split_text(raw_text)
                docs = [Document(page_content=chunk) for chunk in chunks]

                embeddings = get_embeddings_model()
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

            st.success(f"Knowledge base built from {len(uploaded_files)} file(s), {len(chunks)} chunks.")

# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
user_question = st.chat_input("Ask a question about your uploaded material...")

if user_question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload documents and click 'Build Knowledge Base' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved_docs = st.session_state.vectorstore.similarity_search(
                    user_question, k=4
                )
                context = "\n\n".join(d.page_content for d in retrieved_docs)

                prompt = f"""You are a helpful tutor. Use the context below to answer the student's question.
If the answer isn't in the context, say you don't have that information in the uploaded material.

Context:
{context}

Question: {user_question}

Answer:"""

                response = model.generate_content(prompt)
                answer = response.text
                st.markdown(answer)

                with st.expander("Sources used"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})


