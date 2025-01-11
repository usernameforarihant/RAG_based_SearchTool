import streamlit as st
import faiss
import openai
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def generate_embedding(_text: str, api_key: str) -> Optional[np.ndarray]:
    """Generate embeddings with retry logic and proper error handling"""
    if not _text:
        logger.error("Empty text provided for embedding")
        return None
        
    openai.api_key = api_key
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.Embedding.create(
                input=_text,
                model="text-embedding-ada-002"
            )
            embedding = np.array(response['data'][0]['embedding']).reshape(1, -1).astype('float32')
            return embedding
        except Exception as e:
            logger.error(f"Embedding attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return None
            continue
    return None

class EnhancedRAGSystem:
    def __init__(self, api_key: str, index_path: str, data_path: str):
        """Initialize the enhanced RAG system with improved components."""
        self.openai_api_key = api_key
        self.chat_model = "gpt-3.5-turbo-16k"
        
        # Initialize FAISS index
        try:
            self.index = faiss.read_index(index_path)
            logger.info("Successfully loaded FAISS index")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
        
        # Load course data
        self.course_data = self._load_course_data(data_path)

    def _load_course_data(self, data_path: str) -> List[Dict]:
        """Load and parse course data with error handling"""
        try:
            with open(data_path, "r", encoding="utf-8") as file:
                courses = []
                for line in file:
                    if line.strip():
                        try:
                            course_dict = json.loads(line)
                            courses.append(course_dict)
                        except json.JSONDecodeError:
                            courses.append({"content": line.strip()})
            return courses
        except Exception as e:
            logger.error(f"Error loading course data: {e}")
            raise

    def semantic_search(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Tuple[float, Dict]]:
        """Enhanced semantic search with better error handling"""
        if not query:
            logger.warning("Empty query provided")
            return []
            
        query_embedding = generate_embedding(query, self.openai_api_key)
        if query_embedding is None:
            logger.error("Failed to generate embedding for query")
            return []
        
        try:
            distances, indices = self.index.search(query_embedding, k * 2)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.course_data):
                    similarity = 1 / (1 + float(distances[0][i]))
                    if similarity >= threshold:
                        results.append((similarity, self.course_data[idx]))
            
            return sorted(results, key=lambda x: x[0], reverse=True)[:k]
        except Exception as e:
            logger.error(f"Error during semantic search: {str(e)}")
            return []

def main():
    """Main application function with improved error handling"""
    st.set_page_config(
        page_title="Advanced Course Finder",
        page_icon="🎓",
        layout="wide"
    )
    
    # Initialize session state
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'past_queries' not in st.session_state:
        st.session_state.past_queries = []
    
    try:
        # Load configuration
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set it in your .env file.")
            return
        
        # Initialize RAG system
        rag_system = EnhancedRAGSystem(
            api_key=api_key,
            index_path="course_embeddings_openai.index",
            data_path="course_details.txt"
        )
        
        # Main interface
        st.title("🎓 Advanced Course Finder")
        st.write("Search for courses and get personalized recommendations")
        
        # Search interface
        query = st.text_input(
            "What would you like to learn?",
            placeholder="Enter topics, skills, or learning goals..."
        )
        
        col1, col2 = st.columns([4, 1])
        with col2:
            k = st.slider("Number of results", 1, 10, 5)
            threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.6)
        
        if st.button("🔍 Search", use_container_width=True):
            if not query:
                st.warning("Please enter a search query")
                return
                
            with st.spinner("Searching courses..."):
                try:
                    # Perform search
                    results = rag_system.semantic_search(query, k, threshold)
                    
                    if not results:
                        st.warning("No matching courses found. Try adjusting your search terms.")
                        return
                    
                    # Display results
                    st.subheader("📚 Matching Courses")
                    for similarity, course in results:
                        with st.expander(f"Course (Similarity: {similarity:.2%})"):
                            st.write("**Content:**", course.get('content', 'No content available'))
                            st.write("**Topics:**", ', '.join(course.get('topics', ['Not specified'])))
                            st.write("**Difficulty:**", course.get('difficulty', 'Not specified'))
                    
                    # Update history
                    st.session_state.past_queries.append(query)
                    st.session_state.search_history.append({
                        'query': query,
                        'timestamp': datetime.now(),
                        'num_results': len(results)
                    })
                    
                except Exception as e:
                    st.error(f"Error processing search: {str(e)}")
                    logger.error(f"Search error: {str(e)}")
        
        # Show search history in sidebar
        with st.sidebar:
            st.header("📊 Recent Searches")
            for past_query in st.session_state.past_queries[-5:]:
                st.text(f"• {past_query}")
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()