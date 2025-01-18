import streamlit as st
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseAdvisor:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.setup_page_config()
        self.load_styles()
        self.initialize_session_state()
    
    @staticmethod
    def setup_page_config():
        st.set_page_config(
            page_title="Course Advisor AI",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_styles(self):
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
            .sidebar-content {
                background-color: #121212;
                padding: 1.5rem;
                border-radius: 1rem;
                margin-top: 1rem;
                border: 1px solid #333333;
                color: #ffffff;
            }
            .sample-question {
                padding: 0.5rem 1rem;
                background-color: #121212;
                border: 1px solid #333333;
                border-radius: 0.5rem;
                margin-bottom: 0.5rem;
                cursor: pointer;
                color: #ffffff !important;
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
            </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def process_course_line(self, line: str) -> Optional[Dict]:
        """Process a single course line into structured format"""
        try:
            parts = line.split('; ')
            course_dict = {
                'title': '',
                'description': '',
                'curriculum': []
            }
            
            for part in parts:
                if part.startswith('Title:'):
                    course_dict['title'] = part.replace('Title:', '').strip()
                elif part.startswith('Description:'):
                    course_dict['description'] = part.replace('Description:', '').strip()
                elif part.startswith('Curriculum:'):
                    curriculum_text = part.replace('Curriculum:', '').strip()
                    course_dict['curriculum'] = self._parse_curriculum(curriculum_text)
            
            return course_dict
            
        except Exception as e:
            logger.error(f"Error processing course line: {str(e)}")
            return None
    
    def _parse_curriculum(self, curriculum_text: str) -> List[Dict]:
        """Parse curriculum text into structured format"""
        sections = []
        if not curriculum_text:
            return sections
            
        for section in curriculum_text.split('|'):
            if ':' in section:
                name, content = section.split(':', 1)
                topics = content.strip('[]').split(',') if '[' in content else []
                sections.append({
                    'section': name.strip(),
                    'topics': [t.strip() for t in topics if t.strip()]
                })
            else:
                sections.append({
                    'section': section.strip(),
                    'topics': []
                })
        return sections
    
    def load_course_details(self):
        """Load and process course details from file"""
        try:
            with open("course_details.txt", 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            courses = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    course = self.process_course_line(line.strip())
                    if course:  # Only add successfully processed courses
                        courses.append(course)
            
            if not courses:
                st.error("No courses were successfully loaded from the file.")
            return courses
        except Exception as e:
            logger.error(f"Error loading course details: {str(e)}")
            st.error("Failed to load course details. Please check the file.")
            return []

    def _format_course_content(self, course: Dict) -> str:
        """Format course content for document creation"""
        content = f"""
        COURSE DETAILS:

        Title: {course['title']}

        Description: {course['description']}

        Curriculum:
        """
        for section in course['curriculum']:
            content += f"\n- {section['section']}"
            if section['topics']:
                content += f"\n  Topics: {', '.join(section['topics'])}"
        return content

    def initialize_rag_chain(self):
        """Initialize the RAG chain with error handling and caching"""
        @st.cache_resource
        def _create_chain():
            try:
                courses = self.load_course_details()
                if not courses:
                    raise ValueError("No courses loaded")
                
                docs = []
                for course in courses:
                    content = self._format_course_content(course)
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
                
                embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                chain = self._create_chain(vectorstore)
                return chain
                
            except Exception as e:
                logger.error(f"Error initializing RAG chain: {str(e)}")
                st.error("Failed to initialize the course advisor. Please try again later.")
                return None
        
        return _create_chain()
    
    def _create_chain(self, vectorstore: FAISS):
        """Create the RAG chain"""
        template = """You are a course advisor assistant designed to help users find and understand educational programs.
        
        Available Context:
        {context}
        
        Question: {question}
        
        Please provide a detailed response that includes:
        1. Identification of the most relevant courses
        2. For each relevant course, provide:
           - The exact course title
           - The complete course description
           - The full curriculum structure
        3. Explanation of why each course matches the query
        4. Clear recommendations based on the user's needs
        
        Important: Always include the complete title, description, and curriculum sections for each recommended course exactly as they appear in the course details.
        Remember to mention that all courses are available for free on Analytics Vidhya's platform.
        
        Format your response with clear headings and sections for readability."""
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant courses
        )
        
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    def render_header(self):
        st.markdown('<div class="main-header">' +
                   '<h1>🎓 Advanced Course Advisor AI</h1>' +
                   '</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">' +
                   'Your personal AI guide to discovering the perfect AI and Machine Learning courses' +
                   '</div>', unsafe_allow_html=True)
    
    def render_free_courses_banner(self):
        st.markdown("""
            <div class="free-course-banner">
                <h3>🌟 All Recommended Courses Are Free! 🌟</h3>
                <p>Access all courses at no cost on 
                <a href="https://courses.analyticsvidhya.com/pages/all-free-courses" target="_blank">Analytics Vidhya</a></p>
            </div>
        """, unsafe_allow_html=True)
    
    def render_chat_interface(self, rag_chain):
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
    
    def render_sidebar(self):
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
            
            if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    def run(self):
        """Main application loop"""
        self.render_header()
        self.render_free_courses_banner()
        
        rag_chain = self.initialize_rag_chain()
        if not rag_chain:
            return
        
        self.render_chat_interface(rag_chain)
        self.render_sidebar()

if __name__ == "__main__":
    advisor = CourseAdvisor()
    advisor.run()

    
