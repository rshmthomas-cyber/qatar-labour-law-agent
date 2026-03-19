# test.py
print("Step 1: imports...")
import chromadb
import fitz

print("Step 2: opening PDF...")
doc = fitz.open("qatar_labour_law.pdf")
print(f"Pages: {len(doc)}")

print("Step 3: extracting text...")
full_text = ""
for page in doc:
    full_text += page.get_text()
print(f"Characters: {len(full_text)}")

print("Step 4: chunking...")
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
print(f"Chunks: {len(chunks)}")

print("Step 5: ChromaDB...")
client = chromadb.Client()
collection = client.create_collection(name="qatar_labour_law")
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"Stored: {collection.count()} chunks")
print("ALL STEPS PASSED ✅")