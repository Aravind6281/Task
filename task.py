import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import pinecone
import openai

# Initialize embedding model and vector database
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Replace with your preferred model
VECTOR_DB_NAME = 'rag_pipeline'

def initialize_pinecone(api_key, environment):
    pinecone.init(api_key=api_key, environment=environment)
    if VECTOR_DB_NAME not in pinecone.list_indexes():
        pinecone.create_index(VECTOR_DB_NAME, dimension=384)
    return pinecone.Index(VECTOR_DB_NAME)

def crawl_and_scrape(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract main content (adjust selectors as needed)
    content = ' '.join([p.text for p in soup.find_all('p')])
    return content

def segment_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_and_store(content_chunks, metadata, index):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(content_chunks)
    
    for i, (embedding, chunk) in enumerate(zip(embeddings, content_chunks)):
        index.upsert(
            [(f"{metadata['url']}_chunk_{i}", embedding.tolist(), {"text": chunk, **metadata})]
        )

def query_vector_db(query, index, top_k=5):
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0]
    
    results = index.query(query_embedding.tolist(), top_k=top_k, include_metadata=True)
    return results['matches']

def generate_response(prompt, model="gpt-4", api_key="<your_openai_api_key>"):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Example pipeline execution
def main():
    # Initialize Pinecone
    index = initialize_pinecone(api_key="<your_pinecone_api_key>", environment="us-west1-gcp")
    
    # Crawl and scrape
    url = "https://example.com"
    content = crawl_and_scrape(url)
    chunks = segment_text(content)
    
    # Store embeddings
    metadata = {"url": url}
    embed_and_store(chunks, metadata, index)
    
    # User query
    user_query = "What is the main topic of the website?"
    results = query_vector_db(user_query, index)
    
    # Prepare prompt and generate response
    context = "\n".join([match['metadata']['text'] for match in results])
    prompt = f"Context:\n{context}\n\nQuestion:\n{user_query}"
    response = generate_response(prompt)
    
    print("Generated Response:\n", response)

if __name__ == "__main__":
    main()

