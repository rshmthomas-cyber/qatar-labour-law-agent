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

# --- Article-based chunking (the key fix) ---
def chunk_by_article(text):
    """Split text at Article boundaries — respects legal document structure"""
    # Split on Article markers like "Article (1)", "Article (10)" etc
    pattern = r'(?=Article\s*\(\d+\))'
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # If an article is very long, split it further (max 600 words)
        words = part.split()
        if len(words) <= 600:
            chunks.append(part)
        else:
            # Split long articles into overlapping sub-chunks
            for i in range(0, len(words), 500):
                sub = " ".join(words[i:i+550])
                if sub.strip():
                    chunks.append(sub)
    
    return chunks

@st.cache_resource
def load_agent():
    # Try new PDF first, fall back to old one
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

    # Use Article-based chunking
    chunks = chunk_by_article(full_text)
    
    # Fall back to word chunking if Article pattern not found
    if len(chunks) < 10:
        words = full_text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+500])
            chunks.append(chunk)
            i += 450  # 50-word overlap
    
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="qatar_labour_law")
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return collection, len(chunks), pdf_path

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, 'secrets') else ""
    if not api_key:
        api_key = st.text_input("Claude API Key", type="password", placeholder="sk-ant-...")
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

# --- Load ---
with st.spinner("Loading Qatar Labour Law database..."):
    collection, chunk_count, pdf_used = load_agent()

st.caption(f"📄 Loaded {chunk_count} legal sections from {pdf_used}")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Input ---
if question := st.chat_input("Ask a question about Qatar Labour Law..."):

    if not api_key:
        st.warning("⚠️ Please enter your Claude API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the law..."):

            # Retrieve top 5 chunks (increased from 3)
            results = collection.query(
                query_texts=[question],
                n_results=5
            )
            context = "\n\n---\n\n".join(results['documents'][0])

            # Build conversation history
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

            # Sonnet for better reasoning
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