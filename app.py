import streamlit as st
import chromadb
import anthropic
import fitz
import re
import os

st.set_page_config(
    page_title="Qatar Labour Law Assistant",
    page_icon="⚖️",
    layout="centered"
)

# --- Article-based chunking ---
def chunk_by_article(text):
    """Split text at Article boundaries — respects legal document structure"""
    pattern = r'(?=Article\s*\(\d+\))'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        words = part.split()
        if len(words) <= 600:
            chunks.append(part)
        else:
            for i in range(0, len(words), 500):
                sub = " ".join(words[i:i+550])
                if sub.strip():
                    chunks.append(sub)
    return chunks


# --- Smart retrieval: direct Article lookup + semantic search ---
def get_relevant_chunks(collection, question, all_chunks, n_results=5):
    """
    Direct text search for Article number queries.
    Semantic search for everything else.
    Combines both for best results.
    """
    article_match = re.search(r'article\s*\(?(\d+)\)?', question, re.IGNORECASE)

    if article_match:
        article_num = article_match.group(1)
        search_pattern = f"Article ({article_num})"

        # Direct text search — guaranteed to find the right Article
        direct_chunks = []
        for chunk in all_chunks:
            if search_pattern in chunk:
                direct_chunks.append(chunk)

        # Also semantic search for related context
        semantic = collection.query(
            query_texts=[question],
            n_results=3
        )
        semantic_docs = semantic['documents'][0]

        # Combine — direct match first, semantic context after
        combined = direct_chunks.copy()
        for doc in semantic_docs:
            if doc not in combined:
                combined.append(doc)
        return combined[:6]
    else:
        # Regular semantic search for topic-based questions
        results = collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return results['documents'][0]


# --- Load PDF and build ChromaDB ---
@st.cache_resource
def load_agent():
    pdf_options = [
        "Qatar_Labor_Law_As_of_2024_1728321402.pdf",
        "qatar_labour_law.pdf"
    ]

    pdf_path = None
    for name in pdf_options:
        if os.path.exists(name):
            pdf_path = name
            break

    if not pdf_path:
        st.error("PDF file not found. Please ensure the PDF is in the same folder as app.py")
        st.stop()

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Article-based chunking
    chunks = chunk_by_article(full_text)

    # Fall back to word chunking if Article pattern not found
    if len(chunks) < 10:
        words = full_text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+500])
            chunks.append(chunk)
            i += 450

    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="qatar_labour_law")
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    # Return chunks list too — needed for direct Article lookup
    return collection, len(chunks), pdf_path, chunks


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Safely load API key from secrets or sidebar input
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        api_key = st.text_input(
            "Claude API Key",
            type="password",
            placeholder="sk-ant-..."
        )
        st.caption("Your key is never stored or shared.")
    else:
        st.success("API key loaded from secrets ✅")

    st.divider()
    st.markdown("**💡 Sample Questions**")
    sample_questions = [
        "What is the notice period for resignation?",
        "How is end of service gratuity calculated?",
        "What are annual leave entitlements?",
        "Can an employer terminate without notice?",
        "What are working hours during Ramadan?",
        "What happens if employer doesn't pay on time?",
        "What are the rules for overtime pay?",
        "What is Article 47?",
        "What is Article 73?",
    ]
    for q in sample_questions:
        st.markdown(f"- {q}")

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**⚠️ Disclaimer**")
    st.caption("For reference only. Does not constitute legal advice. Consult a legal professional for specific cases.")


# --- Header ---
st.title("⚖️ Qatar Labour Law Assistant")
st.caption("Ask any question about Qatar Labour Law — powered by Claude AI")
st.divider()

# --- Load collection ---
with st.spinner("Loading Qatar Labour Law database..."):
    collection, chunk_count, pdf_used, all_chunks = load_agent()

st.caption(f"📄 Loaded {chunk_count} legal sections from {pdf_used}")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if question := st.chat_input("Ask a question about Qatar Labour Law..."):

    if not api_key:
        st.warning("⚠️ Please enter your Claude API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the law..."):

            # Smart retrieval — direct for Article numbers, semantic for topics
            chunks = get_relevant_chunks(collection, question, all_chunks)
            context = "\n\n---\n\n".join(chunks)

            # Build full conversation history for Claude
            claude_messages = []
            for msg in st.session_state.messages[:-1]:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            claude_messages.append({
                "role": "user",
                "content": f"""You are an expert Qatar Labour Law assistant helping workers and employers in Qatar understand their legal rights and obligations.

INSTRUCTIONS:
- Answer using the provided legal context below
- Always cite the specific Article number(s) you are referencing
- If multiple articles are relevant, cite all of them
- Give complete, thorough answers — do not cut answers short
- Use clear bullet points and structure for readability
- If the exact answer is not in the context, say: "This specific detail may require consulting the full law or a legal professional" — but still share anything relevant you found
- Never say you cannot answer — always try to help with what is available
- For calculations (gratuity, leave pay etc), show the formula clearly

LEGAL CONTEXT:
{context}

QUESTION: {question}"""
            })

            # Claude Sonnet for strong legal reasoning
            claude = anthropic.Anthropic(api_key=api_key)
            response = claude.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2000,
                messages=claude_messages
            )

            answer = response.content[0].text
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
