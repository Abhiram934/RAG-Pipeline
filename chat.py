import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Web Scraping
def scrape_website(url):
    """ Scrapes content from the website URL """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = [p.get_text() for p in paragraphs]
    return text_content

# Step 2: Convert Text to Embeddings
def create_embeddings(text_data, model):
    """ Converts text data to embeddings """
    return model.encode(text_data)

# Step 3: Store Embeddings in FAISS
def create_vector_database(embeddings):
    """ Creates a vector database (FAISS) """
    dimension = embeddings.shape[1]  # embedding dimension (usually 768 for BERT-based models)
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    index.add(embeddings)  # Add embeddings to the index
    return index

# Step 4: Query Handling
def query_vector_database(query, model, index, text_data):
    """ Handles user query and retrieves relevant information """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=3)  # Top 3 closest results
    results = [text_data[i] for i in indices[0]]
    return results

# Step 5: Generate Response Using Retrieved Chunks
def generate_response(query, retrieved_chunks):
    """ Simple response generator based on retrieved chunks """
    context = "\n".join(retrieved_chunks)
    response = f"Based on the following information, here's the answer to your query '{query}':\n\n{context}"
    return response

# Main RAG Pipeline
def rag_pipeline(url, query):
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Step 1: Scrape website content
    text_data = scrape_website(url)
    
    # Step 2: Convert scraped content to embeddings
    embeddings = create_embeddings(text_data, model)
    
    # Step 3: Store embeddings in FAISS
    index = create_vector_database(np.array(embeddings))
    
    # Step 4: Handle user query
    retrieved_chunks = query_vector_database(query, model, index, text_data)
    
    # Step 5: Generate response
    response = generate_response(query, retrieved_chunks)
    return response

# Example Usage
if __name__ == "__main__":
    url = "https://example.com"  # URL to scrape
    query = "What is the return policy?"
    response = rag_pipeline(url, query)
    print(response)
