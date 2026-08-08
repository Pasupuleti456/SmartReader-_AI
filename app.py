import streamlit as st
import fitz
import pdfplumber
import os
import uuid

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
st.markdown("""
<style>
.stApp { background: #f5f7ff; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172033 0%, #202b46 100%);
}
section[data-testid="stSidebar"] * { color: #f8fafc !important; }
section[data-testid="stSidebar"] input {
    background: white !important;
    color: #172033 !important;
    border-radius: 10px !important;
}

.hero {
    text-align: center;
    padding: 12px 0 25px 0;
}
.badge {
    display: inline-block;
    padding: 7px 16px;
    border-radius: 999px;
    background: #eef2ff;
    color: #4f46e5;
    font-weight: 700;
}
.main-title {
    font-size: 3.1rem;
    font-weight: 800;
    color: #1e293b;
    margin: 12px 0 5px 0;
}
.subtitle {
    color: #64748b;
    font-size: 1.05rem;
}
.pill {
    display: inline-block;
    margin: 10px 4px 0 4px;
    padding: 6px 12px;
    border-radius: 999px;
    background: #ffffff;
    color: #475569;
    border: 1px solid #e2e8f0;
}
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px 18px;
    min-height: 155px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(30,41,59,.07);
}
.card-icon { font-size: 30px; }
.card-title {
    color: #1e293b;
    font-size: 19px;
    font-weight: 750;
    margin: 8px 0;
}
.card-text { color: #64748b; line-height: 1.5; }

.section-title {
    color: #1e293b;
    font-size: 1.5rem;
    font-weight: 750;
    margin: 28px 0 5px 0;
}

.answer-box {
    background: white;
    border-left: 5px solid #4f46e5;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 5px 18px rgba(30,41,59,.06);
}
.answer-title {
    color: #4338ca;
    font-size: 18px;
    font-weight: 750;
    margin-bottom: 10px;
}
.stButton > button {
    border-radius: 10px;
    background: #eef2ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
    font-weight: 650;
}
</style>

<div class="hero">
    <div class="main-title">📖 SmartReader AI📖 </div>
    <div class="subtitle">
        Upload your study material, ask questions, and understand documents faster.
    </div>
    <div>
        <span class="pill">📄 Upload</span>
        <span class="pill">🔎 Search</span>
        <span class="pill">🤖 Ask</span>
        <span class="pill">🧠 Learn</span>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">📄</div>
        <div class="card-title">Smart PDF Reading</div>
        <div class="card-text">Extract useful text and tables from your study material.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🔎</div>
        <div class="card-title">Semantic Search</div>
        <div class="card-text">Find relevant information based on meaning, not only keywords.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🤖</div>
        <div class="card-title">AI Answers</div>
        <div class="card-text">Get answers grounded in your uploaded document.</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# STREAMLIT CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="STUDY MATE",
    page_icon="📖",
    layout="wide"
)

st.title("")
st.write(
    "Upload a PDF and ask questions based on its content."
)


# ==========================================================
# HUGGING FACE API KEY
# ==========================================================

hf_api_key = st.sidebar.text_input(
    "✨Activate StudyMate AI✨ ",
    type="password"
)

if not hf_api_key:
    st.warning(
        "Please enter your Hugging Face API key."
    )
    st.stop()

# Correct Hugging Face environment variable
os.environ["HF_TOKEN"] = hf_api_key


# ==========================================================
# SESSION STATE
# ==========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ==========================================================
# CLEAR CURRENT PDF
# ==========================================================

if st.sidebar.button("🗑️ Clear Current PDF"):

    st.session_state.vectorstore = None
    st.session_state.processed_file = None

    # Create a new session/collection
    st.session_state.session_id = str(uuid.uuid4())

    st.rerun()


# ==========================================================
# EMBEDDING MODEL
# ==========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ==========================================================
# CUSTOM EMBEDDING CLASS
# ==========================================================

class SentenceTransformerEmbeddings:

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def embed_query(self, text):

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return embedding.tolist()


embedding_function = SentenceTransformerEmbeddings(
    embedding_model
)


# ==========================================================
# EXTRACT TEXT FROM PDF
# ==========================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    doc = fitz.open(pdf_path)

    for page in doc:

        page_text = page.get_text("text")

        if page_text:
            text += page_text
            text += "\n"

    doc.close()

    return text


# ==========================================================
# EXTRACT TABLES FROM PDF
# ==========================================================

def extract_tables_from_pdf(pdf_path):

    tables = []

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            extracted_table = page.extract_table()

            if extracted_table:

                rows = []

                for row in extracted_table:

                    row = [
                        str(cell) if cell else ""
                        for cell in row
                    ]

                    rows.append(
                        "\t".join(row)
                    )

                tables.append(
                    "\n".join(rows)
                )

    return tables


# ==========================================================
# TEXT CHUNKING
# ==========================================================

def create_chunks(
    text,
    chunk_size=1000,
    overlap=150
):

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        current_length = 0
        end = start

        while (
            end < len(words)
            and current_length + len(words[end]) <= chunk_size
        ):

            current_length += (
                len(words[end]) + 1
            )

            end += 1

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        # Create overlap
        overlap_words = 0
        overlap_length = 0

        i = end - 1

        while (
            i >= start
            and overlap_length < overlap
        ):

            overlap_length += (
                len(words[i]) + 1
            )

            overlap_words += 1

            i -= 1

        start = end - overlap_words

    return chunks


# ==========================================================
# CREATE CHROMA VECTOR DATABASE
# ==========================================================

def create_chroma_vectorstore(
    text_data,
    session_id
):

    # Create chunks
    chunks = create_chunks(
        text_data
    )

    # Check chunks
    if not chunks:

        raise ValueError(
            "No text chunks were created from the PDF."
        )

    # Create LangChain documents
    documents = []

    for i, chunk in enumerate(chunks):

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "session_id": session_id
                }
            )
        )

    # Persistent ChromaDB directory
    persist_directory = "./chroma_db"

    # Unique collection for current session
    collection_name = (
        "pdf_collection_"
        + session_id.replace("-", "")
    )

    # Create Chroma vector store
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    # Store documents
    vectorstore.add_documents(
        documents
    )

    return vectorstore, len(chunks)


# ==========================================================
# RETRIEVE RELEVANT DOCUMENTS
# ==========================================================

def retrieve_relevant_documents(
    vectorstore,
    question,
    top_k=3
):

    results = vectorstore.similarity_search(
        question,
        k=top_k
    )

    return results


# ==========================================================
# HUGGING FACE OPENAI-COMPATIBLE CLIENT
# ==========================================================

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_api_key
)


# ==========================================================
# RAG PROMPT
# ==========================================================

prompt = PromptTemplate(

    input_variables=[
        "context",
        "question"
    ],

    template="""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved PDF context.

Rules:

1. Use only the supplied PDF context.
2. Do not invent information.
3. If the answer is not present in the context, say:

"I could not find the answer in the uploaded PDF."

4. Keep the answer clear and concise.
5. If the question asks for a score, number, name,
   date, project, or other specific information,
   provide the exact information from the context.

Retrieved PDF Context:
{context}

User Question:
{question}

Answer:
"""
)


# ==========================================================
# PDF UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # Unique file identifier
    file_id = (
        uploaded_file.name
        + "_"
        + str(uploaded_file.size)
    )

    # ------------------------------------------------------
    # PROCESS ONLY NEW FILE
    # ------------------------------------------------------

    if st.session_state.processed_file != file_id:

        pdf_path = (
            "uploaded_"
            + st.session_state.session_id
            + ".pdf"
        )

        # --------------------------------------------------
        # SAVE PDF
        # --------------------------------------------------

        with open(
            pdf_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        # --------------------------------------------------
        # EXTRACT TEXT
        # --------------------------------------------------

        with st.spinner(
            "📄 Extracting text from PDF..."
        ):

            extracted_text = (
                extract_text_from_pdf(
                    pdf_path
                )
            )

        # --------------------------------------------------
        # EXTRACT TABLES
        # --------------------------------------------------

        with st.spinner(
            "📊 Extracting tables from PDF..."
        ):

            extracted_tables = (
                extract_tables_from_pdf(
                    pdf_path
                )
            )

        # --------------------------------------------------
        # COMBINE TEXT AND TABLES
        # --------------------------------------------------

        all_text_data = (
            extracted_text
            + "\n"
            + "\n".join(
                extracted_tables
            )
        )

        # --------------------------------------------------
        # CHECK EXTRACTED TEXT
        # --------------------------------------------------

        if not all_text_data.strip():

            st.error(
                "❌ No readable text was found in this PDF."
            )

            st.warning(
                "This may be a scanned/image-based PDF. "
                "OCR is required for scanned PDFs."
            )

            st.stop()

        st.info(
            f"📄 Extracted "
            f"{len(all_text_data):,} characters."
        )

        # --------------------------------------------------
        # CREATE CHROMA DATABASE
        # --------------------------------------------------

        with st.spinner(
            "🧠 Creating embeddings and storing in ChromaDB..."
        ):

            try:

                vectorstore, chunk_count = (
                    create_chroma_vectorstore(
                        all_text_data,
                        st.session_state.session_id
                    )
                )

                # Save vectorstore in session
                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.processed_file = (
                    file_id
                )

                st.success(
                    "✅ PDF processed successfully!"
                )

                st.success(
                    f"🗄️ Stored {chunk_count} chunks in ChromaDB."
                )

            except Exception as e:

                st.error(
                    f"❌ ChromaDB error: {e}"
                )

                st.stop()


# ==========================================================
# QUESTION INPUT
# ==========================================================

user_question = st.text_input(
    "❓ Ask a question about the PDF:"
)


# ==========================================================
# RAG QUESTION ANSWERING
# ==========================================================

if (
    user_question
    and st.session_state.vectorstore
):

    # ------------------------------------------------------
    # STEP 1: RETRIEVAL
    # ------------------------------------------------------

    with st.spinner(
        "🔎 Searching relevant information..."
    ):

        relevant_documents = (
            retrieve_relevant_documents(
                st.session_state.vectorstore,
                user_question,
                top_k=3
            )
        )

    # ------------------------------------------------------
    # CHECK RETRIEVAL
    # ------------------------------------------------------

    if not relevant_documents:

        st.warning(
            "No relevant information was found."
        )

        st.stop()

    # ------------------------------------------------------
    # STEP 2: CREATE CONTEXT
    # ------------------------------------------------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in relevant_documents
        ]
    )

    # ------------------------------------------------------
    # STEP 3: CREATE FINAL PROMPT
    # ------------------------------------------------------

    final_prompt = prompt.format(
        context=context,
        question=user_question
    )

    # ------------------------------------------------------
    # STEP 4: GENERATE ANSWER
    # ------------------------------------------------------

    with st.spinner(
        "🤖 Generating answer..."
    ):

        try:

            response = client.chat.completions.create(

                # Hugging Face automatically selects
                # an available provider for this model.
                model="openai/gpt-oss-120b",

                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],

                temperature=0.2,

                max_tokens=512
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        except Exception as e:

            st.error(
                f"❌ Error generating answer: {e}"
            )

            st.stop()

    # ------------------------------------------------------
    # STEP 5: DISPLAY ANSWER
    # ------------------------------------------------------

    st.write(
        "### 🤖 Answer"
    )

    st.write(
        answer
    )

    # ------------------------------------------------------
    # STEP 6: SHOW RETRIEVED CONTEXT
    # ------------------------------------------------------

    with st.expander(
        "🔎 View retrieved context"
    ):

        for i, doc in enumerate(
            relevant_documents
        ):

            st.write(
                f"### Chunk {i + 1}"
            )

            st.write(
                doc.page_content
            )

            st.caption(
                "Chunk ID: "
                + str(
                    doc.metadata.get(
                        "chunk_id",
                        "N/A"
                    )
                )
            )
