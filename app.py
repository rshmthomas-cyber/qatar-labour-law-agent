import streamlit as st
import anthropic
import fitz
import re
import os

st.set_page_config(
    page_title="Qatar Labour Law Assistant",
    page_icon="⚖️",
    layout="centered"
)

# --- Password Protection ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("⚖️ Qatar Labour Law Assistant")
        st.markdown("Please enter the password to access this tool.")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                correct = st.secrets.get("APP_PASSWORD", "qatar2022")
            except:
                correct = "qatar2024"
            if password == correct:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        st.stop()

check_password()

# --- Article-based chunking ---
def chunk_by_article(text):
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


# --- Tool functions ---
def search_by_topic(query, chunks, n=5):
    """Search chunks by keyword relevance"""
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True)
    return [c for _, c in scored[:n]]

def get_article_by_number(article_num, chunks):
    """Get specific article by number — direct text match"""
    pattern = f"Article ({article_num})"
    results = [c for c in chunks if pattern in c]
    return results[:2] if results else [f"Article ({article_num}) was not found in the provided law document."]

def compare_articles(article_numbers, chunks):
    """Get multiple articles for comparison"""
    results = []
    for num in article_numbers:
        articles = get_article_by_number(num, chunks)
        results.extend(articles)
    return results


# --- Tool definitions for Claude ---
tools = [
    {
        "name": "search_by_topic",
        "description": "Search the Qatar Labour Law by topic or keyword. Use this for general questions about rights, obligations, leave, salary, termination, working hours etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic to search for e.g. 'overtime pay', 'annual leave', 'termination notice', 'gratuity calculation'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_article_by_number",
        "description": "Get a specific Article by its number. Use this when the user asks about a specific Article number like 'what is article 47'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "article_number": {
                    "type": "string",
                    "description": "The Article number e.g. '47', '73', '51'"
                }
            },
            "required": ["article_number"]
        }
    },
    {
        "name": "compare_articles",
        "description": "Get multiple Articles to compare them side by side. Use when user asks to compare two topics or wants to understand differences between rights.",
        "input_schema": {
            "type": "object",
            "properties": {
                "article_numbers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Article numbers to retrieve e.g. ['47', '51', '54']"
                }
            },
            "required": ["article_numbers"]
        }
    }
]


# --- Agentic loop ---
def run_agent(question, chunks, api_key, status_container):
    """Run the agentic loop — Claude decides which tools to use"""
    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": question}]

    system = """You are an expert Qatar Labour Law assistant helping workers and employers in Qatar understand their legal rights and obligations.

You have access to tools to search the Qatar Labour Law document. Always use tools to find relevant Articles before answering.

INSTRUCTIONS:
- Always use tools first to find relevant legal text
- Cite specific Article numbers in your final answer
- Give complete, thorough answers — do not cut answers short
- Use clear bullet points and structure for readability
- For calculations (gratuity, leave pay etc), show the formula clearly
- Never say you cannot answer — always try to help with what is available
- If something is not in the law document, say so clearly"""

    tools_used = []

    # Agentic loop
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            system=system,
            tools=tools,
            messages=messages
        )

        # Claude wants to use a tool
        if response.stop_reason == "tool_use":
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tools_used.append(f"🔍 Searching: *{block.input.get('query', block.input.get('article_number', block.input.get('article_numbers', '')))}*")
                    status_container.markdown("\n".join(tools_used))

                    # Execute the tool
                    if block.name == "search_by_topic":
                        result = search_by_topic(block.input["query"], chunks)
                        result_text = "\n\n---\n\n".join(result)

                    elif block.name == "get_article_by_number":
                        result = get_article_by_number(block.input["article_number"], chunks)
                        result_text = "\n\n---\n\n".join(result)

                    elif block.name == "compare_articles":
                        result = compare_articles(block.input["article_numbers"], chunks)
                        result_text = "\n\n---\n\n".join(result)

                    else:
                        result_text = "Tool not found"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text
                    })

            messages.append({
                "role": "user",
                "content": tool_results
            })

        # Claude has finished — return final answer
        elif response.stop_reason == "end_turn":
            final_answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer = block.text
            return final_answer

        else:
            return "Something went wrong. Please try again."


# --- Load PDF ---
@st.cache_resource
def load_chunks():
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
        st.error("PDF file not found.")
        st.stop()

    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    chunks = chunk_by_article(full_text)

    if len(chunks) < 10:
        words = full_text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+500])
            chunks.append(chunk)
            i += 450

    return chunks, len(chunks), pdf_path


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")

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
        "Compare resignation vs termination notice",
    ]
    for q in sample_questions:
        st.markdown(f"- {q}")

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**⚠️ Disclaimer**")
    st.caption("For reference only. Does not constitute legal advice.")
    st.divider()
    st.caption("🤖 Powered by Agentic RAG + Claude Sonnet")


# --- Header ---
st.title("⚖️ Qatar Labour Law Assistant")
st.caption("Ask any question about Qatar Labour Law — powered by Agentic AI")
st.divider()

# --- Load ---
with st.spinner("Loading Qatar Labour Law database..."):
    all_chunks, chunk_count, pdf_used = load_chunks()

st.caption(f"📄 Loaded {chunk_count} legal sections · Agentic RAG v2")

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
        # Show live tool use status
        status = st.empty()
        status.markdown("🤔 *Thinking...*")

        answer = run_agent(question, all_chunks, api_key, status)

        # Clear status and show answer
        status.empty()
        st.markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })