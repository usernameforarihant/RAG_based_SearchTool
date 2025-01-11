import faiss
import openai
import numpy as np

# Set up OpenAI API key
openai.api_key = "enter api key"
# Parse the file content
file_path = "course_details.txt"
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Extract relevant content (e.g., titles, descriptions, etc.)
data = [line.strip() for line in lines if line.strip()]  # Remove empty lines

# Function to generate embeddings using OpenAI API
def get_openai_embeddings(texts):
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings).astype('float32')

# Generate embeddings for the file content
print("Generating embeddings...")
embeddings = get_openai_embeddings(data)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

# Save the FAISS index for future use
faiss.write_index(index, "course_embeddings_openai.index")

print(f"FAISS index created and saved with {index.ntotal} entries.")

index = faiss.read_index("course_embeddings_openai.index")

# Example query
query = "Learn how to choose the right LLM for business needs."
response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
query_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1).astype('float32')

# Search for the most similar entries
k = 5  # Number of nearest neighbors
distances, indices = index.search(query_embedding, k)

# Print results
print("Top similar courses:")
for idx in indices[0]:
    print(data[idx])
