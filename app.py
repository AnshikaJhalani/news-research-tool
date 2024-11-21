from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from utils import fetch_content_from_urls,create_vector_store, query_vector_store
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
app = Flask(__name__)

# Initialize embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if FAISS index exists
try:
    # Try loading an existing FAISS index
    vector_store = FAISS.load_local("faiss_index", embedding_model)
    print("FAISS index loaded successfully.")
except Exception as e:
    # If no index exists, initialize an empty FAISS vector store
    print("FAISS index not found. Initializing an empty index...")
    from langchain.schema import Document
    empty_documents = [Document(page_content="This is a placeholder document", metadata={"source": "init"})]
    vector_store = FAISS.from_documents(empty_documents, embedding_model)
    vector_store.save_local("faiss_index") 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_urls', methods=['POST'])
def load_articles():
    data = request.json
    urls = data.get('urls', [])
    documents = fetch_content_from_urls(urls)

    if documents:
        try:
            create_vector_store(documents, vector_store, embedding_model)
            return jsonify({"message": "Articles processed and indexed successfully."})
        except Exception as e:
            print(f"Error while creating vector store: {e}")
            return jsonify({"error": f"Failed to create vector store: {str(e)}"}), 500
    else:
        return jsonify({"error": "No documents were fetched from the URLs provided."}), 400

@app.route('/query', methods=['POST'])
def query_bot():
    data = request.json
    question = data.get('question', "")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        docs = vector_store.similarity_search(question, k=1) if vector_store else []
        if docs:
            # Get the content and source URL from the retrieved document
            document_content = docs[0].page_content
            source_url = docs[0].metadata.get("source", "No source available")

            # Use the QA pipeline to extract the answer
            answer = qa_pipeline(question=question, context=document_content)

            return jsonify({
                "answer": answer["answer"],  # Extracted answer
                "context": source_url        # Source URL of the answer
            })
        else:
            return jsonify({
                "answer": "No relevant information found.",
                "context": "No source available."
            })
    except Exception as e:
        print(f"Error querying vector store: {str(e)}")
        return jsonify({"error": f"Error querying vector store: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
