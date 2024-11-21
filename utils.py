from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup

def fetch_content_from_urls(urls):
    """
    Fetches content from a list of URLs and converts it into Document objects.
    Each Document includes page content and metadata (source URL).
    """
    documents = []
    for url in urls:
        try:
            print(f"Fetching URL: {url}")
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP request errors
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract content (adjust selectors based on site structure)
            paragraphs = soup.find_all("p")
            content = "\n".join([para.get_text() for para in paragraphs])
            
            if content.strip():
                documents.append(Document(page_content=content.strip(), metadata={"source": url}))
            else:
                print(f"No usable content found for URL: {url}")
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
    return documents

def create_vector_store(documents, vector_store, embedding_model):
    """
    Splits documents into chunks and adds them to the FAISS vector store.
    """
    # Ensure all items in `documents` are `Document` objects
    if not all(isinstance(doc, Document) for doc in documents):
        raise ValueError("All items in `documents` must be `Document` objects.")
    
    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    if not split_docs:
        raise ValueError("No chunks were created from the documents.")

    # Add chunks to the vector store
    vector_store.add_documents(split_docs)
    print("Documents successfully added to the vector store.")

def query_vector_store(question, vector_store, embedding_model):
    """
    Queries the FAISS vector store and returns the most relevant document's content and metadata.
    """
    try:
        # Perform similarity search for the top 1 result
        docs = vector_store.similarity_search(question, k=1)

        if docs:
            # Extract the top document's content and source URL
            document_content = docs[0].page_content
            source_url = docs[0].metadata.get("source", "No source available")
            return document_content, source_url
        else:
            return "No relevant context found.", "No source available"
    except Exception as e:
        print(f"Error querying vector store: {e}")
        raise
