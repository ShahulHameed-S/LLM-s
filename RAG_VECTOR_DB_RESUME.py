import os
import fitz  # PyMuPDF
import chromadb
import ollama

# ---------------------------------------------------------
# Step 1: Extract Text from PDF
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_path, filename):
    doc = fitz.open(pdf_path)
    text = f"--- Source Document: {filename} ---\n" 
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# ---------------------------------------------------------
# Step 2: Chunk the Text
# ---------------------------------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

# ---------------------------------------------------------
# Step 3: Process Entire Folder
# ---------------------------------------------------------
def process_pdf_folder(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Reading: {filename}...")
            
            pdf_content = extract_text_from_pdf(pdf_path, filename)
            document_chunks = chunk_text(pdf_content)
            all_chunks.extend(document_chunks)
            
    return all_chunks

# ---------------------------------------------------------
# Step 4: Setup ChromaDB and Embed
# ---------------------------------------------------------
# Updated to your specific local path
folder_location = r"C:\shahul\Desktop\Resume"  

print(f"Scanning folder: {folder_location}")
document_chunks = process_pdf_folder(folder_location)

if not document_chunks:
    print("No PDFs found or no text could be extracted. Exiting.")
    exit()

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="pdf_collection")

collection.add(
    documents=document_chunks,
    ids=[f"chunk_{i}" for i in range(len(document_chunks))]
)
print(f"Added a total of {len(document_chunks)} chunks to the vector database.")

# ---------------------------------------------------------
# Step 5: Query and Retrieve
# ---------------------------------------------------------
user_query = "tell me about Rachith V?"

results = collection.query(
    query_texts=[user_query],
    n_results=3
)

retrieved_chunks = results['documents'][0]
context = "\n---\n".join(retrieved_chunks)

# ---------------------------------------------------------
# Step 6: Generate Answer with Ollama
# ---------------------------------------------------------
prompt = f"""Answer the question based ONLY on the context provided below. 
If the answer is not in the context, say "I cannot answer this based on the PDF."

Context:
{context}

Question: {user_query}
Answer:"""

print(f"\nAsking your local model: '{user_query}'\nThinking...")

# Updated to use phi4-mini:latest
response = ollama.chat(
    model='phi3:latest',  
    messages=[{'role': 'user', 'content': prompt}]
)

print("\nFinal Answer:\n")
print(response['message']['content'])