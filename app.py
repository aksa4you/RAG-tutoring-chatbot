import streamlit as st
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pypdf import PdfReader

st.set_page_config(page_title="RAG Tutoring Chatbot", page_icon="📚")

st.title("📚 RAG-Based Tutoring Chatbot")
st.caption("Upload your study material (PDF or text), then ask questions about it.")

# ---------- Configure Gemini ----------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
if not GOOGLE_API_KEY:
    st.error("Gemini API key not found. Add it in Settings → Secrets as GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


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


