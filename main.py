import streamlit as st
import openai
import faiss
import os
import numpy as np

# Page configuration and styling
st.set_page_config(
    page_title="Course Advisor Bot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem !important;
        padding-bottom: 1rem;
    }
    .stSubheader {
        color: #34495e;
        font-size: 1.5rem !important;
        padding-top: 1rem;
    }
    .course-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the FAISS index
@st.cache_resource
def load_faiss_index():
    return faiss.read_index("course_embeddings_openai.index")

# Load course data
@st.cache_data
def load_course_data():
    try:
        with open("course_details.txt", "r", encoding="utf-8") as f:
            details = f.read().splitlines()
        with open("course_links.txt", "r", encoding="utf-8") as f:
            links = f.read().splitlines()
        return details, links
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return [], []

def process_context(retrieved_courses, max_tokens=3000):
    """Better context processing for more relevant responses"""
    total_tokens = 0
    processed_context = []
    
    for course, link, similarity_score in retrieved_courses:
        # Rough token estimation (4 chars ~ 1 token)
        estimated_tokens = len(course) // 4
        
        if total_tokens + estimated_tokens > max_tokens:
            break
            
        processed_context.append({
            'content': course,
            'link': link,
            'relevance': 1 - similarity_score  # Convert distance to similarity
        })
        total_tokens += estimated_tokens
    
    # Sort by relevance and format context
    processed_context.sort(key=lambda x: x['relevance'], reverse=True)
    return "\n\n".join([f"Course (Relevance: {c['relevance']:.2f}):\n{c['content']}\nLink: {c['link']}" 
                       for c in processed_context])

def retrieve_course(query, top_k=3):
    """Retrieve the most relevant courses based on the query."""
    try:
        # Ensure top_k doesn't exceed the number of available courses
        max_courses = min(top_k, len(course_details))
        
        query_embedding = openai.Embedding.create(
            input=query, 
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(query_vector, max_courses)
        
        # Validate indices
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if 0 <= i < len(course_details):  # Check if index is valid
                results.append((course_details[i], course_links[i], float(dist)))
        
        return results
    except Exception as e:
        st.error(f"Error during course retrieval: {str(e)}")
        return []

def generate_enhanced_response(context, query):
    """Generate more structured and informative responses"""
    try:
        messages = [
            {"role": "system", "content": """You are a helpful course advisor. Structure your responses as follows:
                1. Direct answer to the question
                2. Key highlights from relevant courses
                3. Specific course recommendations
                Always be concise and specific."""},
            {"role": "user", "content": f"""Context: {context}
                
                Question: {query}
                
                Please provide:
                1. A direct answer
                2. Key relevant points from the courses
                3. Specific course recommendations"""}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error generating the response."

# Initialize
index = load_faiss_index()
course_details, course_links = load_course_data()

# Validate data
if not course_details or not course_links:
    st.error("Failed to load course data. Please check your data files.")
    st.stop()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Sidebar
with st.sidebar:
    st.title("About")
    st.info("""
    This Course Advisor Bot helps you find relevant courses and answers your questions about them. 
    Simply type your question in the search box and get personalized recommendations! 🎯
    """)
    
    st.subheader("How to use")
    st.markdown("""
    1. Type your question in the search box
    2. Wait for the AI to analyze your query
    3. Get personalized course recommendations
    4. Click on course links to learn more
    """)
    
    # Add filter options
    st.subheader("Search Settings")
    top_k = st.slider("Number of courses to retrieve", min_value=1, max_value=5, value=3)

# Main content
st.title("🎓 Course Advisor Bot")
st.markdown("### Your AI-powered course recommendation assistant")

# Display total number of courses
st.info(f"📚 Total courses available: {len(course_details)}")

# Create two columns for search
col1, col2 = st.columns([4, 1])

# Search box with placeholder
with col1:
    query = st.text_input(
        "",
        placeholder="Ask anything about courses (e.g., 'What courses teach machine learning?')",
        key="query"
    )

# Add a search button
with col2:
    search_button = st.button("🔍 Search")

if query and search_button:
    with st.spinner("🤔 Analyzing your question..."):
        retrieved_courses = retrieve_course(query, top_k=top_k)
        
        if retrieved_courses:
            # Use enhanced context processing
            context = process_context(retrieved_courses)
            
            # Generate and display response
            answer = generate_enhanced_response(context, query)
            
            # Display the answer in a nice format
            st.markdown("### 💡 Answer")
            st.write(answer)
            
            # Display courses in a more organized way
            st.markdown("### 📚 Recommended Courses")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Detailed View", "Compact View"])
            
            with tab1:
                for i, (course, link, similarity) in enumerate(retrieved_courses, 1):
                    with st.expander(f"Course {i}: {course[:100]}..."):
                        st.markdown(f"**Course Details:**")
                        st.write(course)
                        st.markdown(f"**Relevance Score**: {((1-similarity) * 100):.1f}%")
                        st.markdown(f"**Course Link**: [{link}]({link})")
            
            with tab2:
                for i, (course, link, similarity) in enumerate(retrieved_courses, 1):
                    st.markdown(f"**{i}. [{course[:100]}...]({link})**")
                    st.markdown(f"Relevance: {((1-similarity) * 100):.1f}%")
        else:
            st.warning("⚠️ No relevant courses found or an error occurred.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ by Your Course Advisor Bot</p>
    </div>
    """,
    unsafe_allow_html=True
)
