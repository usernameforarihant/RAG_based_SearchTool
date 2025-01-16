import streamlit as st
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Course Advisor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Dark Mode
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #4a90e2);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .sub-header {
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #1e1e1e;
        border-left: 4px solid #4a9eff;
        color: #ffffff;
    }
    .chat-message.assistant {
        background-color: #282828;
        border-left: 4px solid #7c4dff;
        color: #ffffff;
    }
    .course-card {
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #333333;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .course-card:hover {
        transform: translateY(-2px);
        border-color: #4a9eff;
    }
    .course-title {
        color: #4a9eff;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .curriculum-section {
        margin-top: 0.8rem;
        padding-left: 1.2rem;
        border-left: 2px solid #333333;
        color: #ffffff;
    }
    .sidebar-content {
        background-color: #121212;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-top: 1rem;
        border: 1px solid #333333;
        color: #ffffff;
    }
    .sidebar-content h2,
    .sidebar-content h3 {
        color: #4a9eff;
        margin-bottom: 1rem;
    }
    .sidebar-content p,
    .sidebar-content li {
        color: #ffffff;
    }
    .sidebar-content ul {
        list-style-type: none;
        padding-left: 0;
    }
    .sidebar-content li {
        margin-bottom: 0.5rem;
        padding-left: 1.2rem;
        position: relative;
    }
    .sidebar-content li:before {
        content: "•";
        color: #4a9eff;
        position: absolute;
        left: 0;
    }
    .free-course-banner {
        background-color: #282828;
        color: #00e676;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid #333333;
    }
    .free-course-banner a {
        color: #4a9eff;
        text-decoration: none;
    }
    .free-course-banner a:hover {
        text-decoration: underline;
    }
    .sample-question {
        padding: 0.5rem 1rem;
        background-color: #121212;
        border: 1px solid #333333;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: background-color 0.2s;
        color: #ffffff !important;
        width: 100%;
        text-align: left;
    }
    .sample-question:hover {
        background-color: #282828;
        border-color: #4a9eff;
    }
    .stButton>button {
        background-color: #121212;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #282828;
        border-color: #4a9eff;
    }
    </style>
""", unsafe_allow_html=True)

# Rest of the functions remain the same
def process_course_line(line):
    """Process a single course line into structured format"""
    try:
        parts = line.split('; ')
        course_dict = {}
        
        for part in parts:
            if part.startswith('Title:'):
                course_dict['title'] = part.replace('Title:', '').strip()
            elif part.startswith('Description:'):
                course_dict['description'] = part.replace('Description:', '').strip()
            elif part.startswith('Curriculum:'):
                curriculum_text = part.replace('Curriculum:', '').strip()
                sections = []
                if curriculum_text:
                    for section in curriculum_text.split('|'):
                        if ':' in section:
                            name, content = section.split(':', 1)
                            topics = content.strip('[]').split(',') if '[]' in content else []
                            sections.append({
                                'section': name.strip(),
                                'topics': [t.strip() for t in topics if t.strip()]
                            })
                        else:
                            sections.append({
                                'section': section.strip(),
                                'topics': []
                            })
                course_dict['curriculum'] = sections
        
        return course_dict
    except Exception as e:
        st.error(f"Error processing course line: {str(e)}")
        return None

def load_course_details():
    """Load and process course details"""
    try:
        with open("course_details.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        courses = []
        for line in lines:
            if line.strip():  # Skip empty lines
                course = process_course_line(line.strip())
                if course:  # Only add successfully processed courses
                    courses.append(course)
        
        if not courses:
            st.error("No courses were successfully loaded from the file.")
        return courses
    except Exception as e:
        st.error(f"Error loading course details: {str(e)}")
        return []

def initialize_rag_chain():
    """Initialize the RAG chain components"""
    courses = load_course_details()
    docs = []
    for course in courses:
        content = f"""
        Course: {course['title']}
        
        Description:
        {course['description']}
        
        Curriculum:
        """
        for section in course['curriculum']:
            content += f"\n{section['section']}"
            if section['topics']:
                content += f": {', '.join(section['topics'])}"
        
        doc = Document(
            page_content=content,
            metadata={"title": course['title']}
        )
        docs.append(doc)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=80
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chromadb"
    )
    retriever = vectorstore.as_retriever()
    
    template = """You are a course advisor assistant designed to help users find and understand educational programs based on their specific needs.
    
    Available Context:
    {context}
    
    Question: {question}
    
    Please provide a detailed response that:
    1. Identifies relevant courses from the context
    2. Explains why each course matches the query
    3. Highlights key aspects of each course (title, description, curriculum)
    4. Makes clear recommendations based on the user's needs
    
    Format your response in a clear, structured way with appropriate headings and bullet points when listing features or curriculum items.
    Remember to mention that all courses are available for free on Analytics Vidhya's platform."""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    # Main header with gradient background
    st.markdown('<div class="main-header">' +
                '<h1>🎓 Advanced Course Advisor AI</h1>' +
                '</div>', unsafe_allow_html=True)
    
    # Subheader with value proposition
    st.markdown('<div class="sub-header">' +
                'Your personal AI guide to discovering the perfect AI and Machine Learning courses' +
                '</div>', unsafe_allow_html=True)
    
    # Free courses banner
    st.markdown("""
        <div class="free-course-banner">
            <h3>🌟 All Recommended Courses Are Free! 🌟</h3>
            <p>Access all courses at no cost on 
            <a href="https://courses.analyticsvidhya.com/pages/all-free-courses" target="_blank">Analytics Vidhya</a></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG chain
    rag_chain = initialize_rag_chain()
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask me about courses... (e.g., 'What courses cover RAG systems?')"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing courses..."):
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
    
    # Enhanced Sidebar with Dark Mode
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-content">
                <h2>💡 Quick Start Guide</h2>
                <p>Ask me anything about our courses! I can help you:</p>
                <ul>
                    <li>Find courses matching your interests</li>
                    <li>Compare different courses</li>
                    <li>Explore course curricula</li>
                    <li>Get personalized recommendations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class="sidebar-content">
                <h3>📝 Sample Questions</h3>
                <p>Click any question to get started:</p>
            </div>
        """, unsafe_allow_html=True)
        
        sample_questions = [
            "What courses cover RAG systems?",
            "Tell me about LLM selection courses",
            "What are the key topics in the RAG course?",
            "Compare the available courses",
            "Which course is best for beginners?",
            "What advanced courses do you recommend?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=question, use_container_width=True):
                st.chat_input_placeholder = question
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
