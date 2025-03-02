import os
import numpy as np
import faiss
import PyPDF2
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512, truncation=True)

# Initialize FAISS index
index = None
chunks = []

# -------------------------------
# Utility Functions
# -------------------------------
def extract_text_from_file(filename):
    """Extract text from PDF or TXT file."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        text = ""
        with open(filename, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    else:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunk_list = []
    i = 0
    while i < len(words):
        chunk = words[i: i + chunk_size]
        chunk_list.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunk_list

def build_faiss_index(chunks):
    """Generate embeddings and create FAISS index."""
    global index
    chunk_embeddings = embedder.encode(chunks, show_progress_bar=True)
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    
    embedding_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(chunk_embeddings)

# -------------------------------
# Flask API Endpoints
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles document upload and processing."""
    global chunks, index
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Process document
    document_text = extract_text_from_file(file_path)
    if not document_text.strip():
        return jsonify({"error": "No text extracted from document"}), 400

    chunks = chunk_text(document_text)
    build_faiss_index(chunks)

    return jsonify({"message": f"Document processed successfully with {len(chunks)} chunks"}), 200

@app.route("/ask", methods=["POST"])
def ask_question():
    """Handles user queries and returns answers."""
    if not index or len(chunks) == 0:
        return jsonify({"error": "No document uploaded"}), 400

    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Retrieve relevant chunks
    q_embedding = embedder.encode([question])
    q_embedding = np.array(q_embedding, dtype=np.float32)
    q_embedding = q_embedding / np.linalg.norm(q_embedding, axis=1, keepdims=True)
    
    scores, indices = index.search(q_embedding, k=3)
    relevant_chunks = [chunks[idx] for idx in indices[0]]

    if scores[0] < 0.35:
        return jsonify({"answer": "No relevant information found in the document"}), 200

    context = " ".join(relevant_chunks)
    context = context[:1000]  # Truncate to avoid exceeding model input limits

    # Generate answer
    prompt = f"Based on the following context:\n{context}\n\nQuestion: {question}\n\nProvide a detailed explanation."
    result = gen_pipeline(prompt, do_sample=False, num_return_sequences=1)[0]['generated_text']

    return jsonify({"answer": result.strip()}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
