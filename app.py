import streamlit as st
import chromadb
import anthropic
import fitz

# --- Page Config ---
st.set_page_config(
    page_title="Qatar Labour Law Assistant",
    page_icon="⚖️",
    layout="centered"
)

# --- Load and index PDF ---
@st.cache_resource
def load_agent():
    doc = fitz.open("qatar_labour_law.pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    def chunk_text(text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    chunks = chunk_text(full_text)
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="qatar_labour_law")
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input(
        "Claude API Key",
        type="password",
        placeholder="sk-ant-..."
    )
    st.caption("Your key is never stored or shared.")
    st.divider()
    st.markdown("**💡 Sample Questions**")
    st.markdown("- What is the notice period for resignation?")
    st.markdown("- How is end of service gratuity calculated?")
    st.markdown("- What are annual leave entitlements?")
    st.markdown("- Can employer terminate without notice?")
    st.markdown("- What are working hours during Ramadan?")
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("**⚠️ Disclaimer**")
    st.caption("This tool is for reference only and does not constitute legal advice.")

# --- Header ---
st.title("⚖️ Qatar Labour Law Assistant")
st.caption("Ask any question about Qatar Labour Law — powered by Claude AI")
st.divider()

# --- Load collection ---
with st.spinner("Loading Qatar Labour Law database..."):
    collection = load_agent()

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display full conversation history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if question := st.chat_input("Ask a question about Qatar Labour Law..."):

    if not api_key:
        st.warning("⚠️ Please enter your Claude API key in the sidebar.")
        st.stop()

    # Show and store user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the law..."):

            # Retrieve relevant chunks
            results = collection.query(
                query_texts=[question],
                n_results=3
            )
            context = "\n\n".join(results['documents'][0])

            # Build full message history for Claude
            # This gives Claude memory of the whole conversation
            claude_messages = []
            
            # Add previous exchanges
            for msg in st.session_state.messages[:-1]:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current question with context
            claude_messages.append({
                "role": "user",
                "content": f"""You are a Qatar Labour Law assistant helping workers and employers understand their rights.
                
Answer using ONLY the context below. If not found, say "This specific detail may not be covered in the provided sections. Please consult a legal professional."

Context from Qatar Labour Law:
{context}

Question: {question}

Answer clearly, use bullet points where helpful, and always cite the Article number if mentioned."""
            })

            # Call Claude with full history
            claude = anthropic.Anthropic(api_key=api_key)
            response = claude.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                messages=claude_messages
            )

            answer = response.content[0].text
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })