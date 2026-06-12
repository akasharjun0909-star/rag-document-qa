import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Document Q&A Assistant")
st.markdown("Upload a PDF document and ask any question about its contents.")

# ── Upload Section ──
st.subheader("Step 1 — Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Indexing document... please wait"):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )
    if response.status_code == 200:
        st.success(f"✅ {response.json()['message']}")
        st.session_state["doc_ready"] = True
    else:
        st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
        st.session_state["doc_ready"] = False

# ── Question Section ──
st.subheader("Step 2 — Ask a question")
question = st.text_input("Type your question here", placeholder="e.g. What is the main conclusion of this document?")

if st.button("Ask", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question}
            )
        if response.status_code == 200:
            data = response.json()
            st.markdown("### Answer")
            st.success(data["answer"])
        else:
            st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

# ── Sidebar ──
with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
    1. Upload any PDF document
    2. The app splits it into chunks
    3. Gemini AI embeds each chunk
    4. FAISS stores them as vectors
    5. Your question retrieves the most relevant chunks
    6. Gemini generates an accurate answer
    """)
    st.markdown("---")
    st.markdown("Built with LangChain · FAISS · FastAPI · Streamlit · Gemini")