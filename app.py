
# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain.docstore.document import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import tiktoken

# def init_session_state():
#     """Initialize session state variables"""
#     if 'openai_api_key' not in st.session_state:
#         st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY")
#     if 'messages' not in st.session_state:
#         st.session_state['messages'] = []
#     if 'rag_chain' not in st.session_state:
#         st.session_state['rag_chain'] = None

# def load_course_data(file_path="course_details.txt"):
#     """Load and process course details with robust encoding handling"""
#     try:
#         # First try UTF-8
#         with open(file_path, 'r', encoding='utf-8') as file:
#             text = file.read()
#     except UnicodeDecodeError:
#         try:
#             # If UTF-8 fails, try with utf-8-sig (handles BOM)
#             with open(file_path, 'r', encoding='utf-8-sig') as file:
#                 text = file.read()
#         except UnicodeDecodeError:
#             # If both fail, try with latin-1
#             with open(file_path, 'r', encoding='latin-1') as file:
#                 text = file.read()
    
#     return [Document(page_content=text, metadata={"source": file_path})]

# def load_course_links(file_path="course_links.txt"):
#     """Load course links from file"""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             links = file.read().splitlines()
#         return links
#     except Exception as e:
#         st.error(f"Error loading course links: {str(e)}")
#         return []
    
# def get_prompt_template():
#     """Return the prompt template for the RAG system"""
#     template = """You are an expert course advisor who helps users find the most relevant courses based on their interests and needs.

# Available Context:
# {context}

# User Question: {question}

# IMPORTANT: You MUST analyze and present information about ALL courses in the context that are relevant to the query. Make sure to:

# 1. Review ALL courses in the provided context thoroughly
# 2. Present EACH relevant course separately and clearly
# 3. For EACH course, provide:
#    - Title (exactly as shown in the context)
#    - Complete description
#    - Full curriculum breakdown
#    - Key highlights and unique features
#    - Specific relevance to the query

# Format the response with clear separation between courses using headers.

# If multiple courses are found:
# - Present them in order of relevance to the query
# - Explain how they complement or differ from each other
# - Help the user understand the unique value of each course

# If only one course is found:
# - Explicitly mention that it's the only relevant course found
# - Explain why other courses might not be relevant

# Remember: Use ONLY information from the provided context. If you're not showing other courses, it should be because they're not in the context or not relevant, not because of processing limitations."""
    
#     return ChatPromptTemplate.from_template(template)
# def initialize_rag_chain():
#     """Initialize the RAG pipeline with improved retrieval"""
#     try:
#         # Load documents and links
#         docs = load_course_data()
#         links = load_course_links()
        
#         # Add links to document metadata
#         for i, doc in enumerate(docs):
#             doc.metadata["links"] = links
        
#         # Modified text splitter with larger chunk size and overlap
#         text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             chunk_size=1500,
#             chunk_overlap=200,
#             separators=["\nTitle:", "\n\nTitle:", "Title:", ";"]
#         )
#         splits = text_splitter.split_documents(docs)
        
#         # Add links to split documents' metadata
#         for split in splits:
#             split.metadata["links"] = links
        
#         # Initialize embeddings and vector store
#         embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['openai_api_key'])
#         vectorstore = Chroma.from_documents(
#             documents=splits,
#             embedding=embeddings,
#             persist_directory="./chromadb"
#         )
        
#         # Enhanced retriever configuration
#         retriever = vectorstore.as_retriever(
#             search_type="mmr",
#             search_kwargs={
#                 "k": 8,
#                 "fetch_k": 20,
#                 "lambda_mult": 0.7
#             }
#         )
        
#         # Initialize LLM
#         llm = ChatOpenAI(
#             model="gpt-4-turbo-preview",
#             temperature=0.2,
#             max_tokens=4000,
#             openai_api_key=st.session_state['openai_api_key']
#         )
        
#         # Create and return RAG chain
#         return (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | get_prompt_template()
#             | llm
#             | StrOutputParser()
#         )
    
#     except Exception as e:
#         st.error(f"Error initializing RAG chain: {str(e)}")
#         return None

# # Update the get_prompt_template() function
# def get_prompt_template():
#     """Return the prompt template for the RAG system"""
#     template = """You are an expert course advisor who helps users find the most relevant courses based on their interests and needs.

# Available Context:
# {context}

# User Question: {question}

# IMPORTANT: You MUST analyze and present information about ALL courses in the context that are relevant to the query. Make sure to:

# 1. Review ALL courses in the provided context thoroughly
# 2. Present EACH relevant course separately and clearly
# 3. For EACH course, provide:
#    - Title (exactly as shown in the context)
#    - Course Link (provide the corresponding link from metadata["links"] list based on the course order)
#    - Complete description
#    - Full curriculum breakdown
#    - Key highlights and unique features
#    - Specific relevance to the query

# Format the response with clear separation between courses using headers and make the title a clickable link using the corresponding course link.

# If multiple courses are found:
# - Present them in order of relevance to the query
# - Explain how they complement or differ from each other
# - Help the user understand the unique value of each course
# - Include the course link for each course

# If only one course is found:
# - Explicitly mention that it's the only relevant course found
# - Explain why other courses might not be relevant
# - Include the course link

# Remember: Use ONLY information from the provided context. If you're not showing other courses, it should be because they're not in the context or not relevant, not because of processing limitations.

# Format each course title as a clickable link, for example:
# # [Course Title](course_link)"""
#     return ChatPromptTemplate.from_template(template)


# def create_sidebar():
#     """Create the sidebar with information and tips"""
#     with st.sidebar:
#         st.header("About")
#         st.write("""
#         This chatbot helps you discover courses based on your interests and needs. 
#         It provides detailed information about:
#         - Course content and curriculum
#         - Key highlights
#         - Why it's relevant to your interests
#         """)
        
#         st.header("Tips")
#         st.write("""
#         - Be specific about your interests
#         - Ask about particular topics or skills
#         - Compare different courses
#         - Ask for recommendations based on your level
#         """)
        
#         # Add reset button
#         if st.button("Reset Chat"):
#             st.session_state['messages'] = []
#             st.session_state['rag_chain'] = initialize_rag_chain()
#             st.rerun()

# def main():
#     """Main function to run the Streamlit app"""
#     # Load environment variables
#     load_dotenv()
    
#     # Set page config
#     st.set_page_config(
#         page_title="Course Recommendation Chatbot",
#         page_icon="🎓",
#         layout="wide"
#     )
    
#     # Initialize session state
#     init_session_state()
    
#     # Create main UI
#     st.title("🎓 Course Recommendation Chatbot")
#     st.write("Ask me about our courses! I'll help you find the perfect learning path.")
    
#     # Initialize RAG chain if not already initialized
#     if st.session_state['rag_chain'] is None:
#         with st.spinner("Initializing the course advisor..."):
#             st.session_state['rag_chain'] = initialize_rag_chain()
    
#     # Create sidebar
#     create_sidebar()
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if query := st.chat_input("What courses are you interested in?"):
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(query)
#         st.session_state.messages.append({"role": "user", "content": query})
        
#         # Generate and display response
#         with st.chat_message("assistant"):
#             with st.spinner("Searching for relevant courses..."):
#                 if st.session_state['rag_chain'] is not None:
#                     try:
#                         response = st.session_state['rag_chain'].invoke(query)
#                         st.markdown(response)
#                         st.session_state.messages.append({"role": "assistant", "content": response})
#                     except Exception as e:
#                         st.error(f"Error generating response: {str(e)}")
#                 else:
#                     st.error("System not properly initialized. Please try refreshing the page.")

# if __name__ == "__main__":
#     main()
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
import tiktoken

def init_session_state():
    """Initialize session state variables"""
    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY")
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'rag_chain' not in st.session_state:
        st.session_state['rag_chain'] = None

def load_course_data(file_path="course_details.txt"):
    """Load and process course details with robust encoding handling"""
    try:
        # First try UTF-8
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try with utf-8-sig (handles BOM)
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                text = file.read()
        except UnicodeDecodeError:
            # If both fail, try with latin-1
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
    
    return [Document(page_content=text, metadata={"source": file_path})]

def load_course_links(file_path="course_links.txt"):
    """Load course links from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            links = file.read().splitlines()
        return links
    except Exception as e:
        st.error(f"Error loading course links: {str(e)}")
        return []
    
def get_prompt_template():
    """Return the prompt template for the RAG system"""
    template = """You are an expert course advisor who helps users find the most relevant courses based on their interests and needs.

Available Context:
{context}

User Question: {question}

IMPORTANT: You MUST analyze and present information about ALL courses in the context that are relevant to the query. Make sure to:

1. Review ALL courses in the provided context thoroughly
2. Present EACH relevant course separately and clearly
3. For EACH course, provide:
   - Title (exactly as shown in the context)
   - Course Link (provide the corresponding link from metadata["links"] list based on the course order)
   - Complete description
   - Full curriculum breakdown
   - Key highlights and unique features
   - Specific relevance to the query

Format the response with clear separation between courses using headers and make the title a clickable link using the corresponding course link.

If multiple courses are found:
- Present them in order of relevance to the query
- Explain how they complement or differ from each other
- Help the user understand the unique value of each course
- Include the course link for each course

If only one course is found:
- Explicitly mention that it's the only relevant course found
- Explain why other courses might not be relevant
- Include the course link

Remember: Use ONLY information from the provided context. If you're not showing other courses, it should be because they're not in the context or not relevant, not because of processing limitations.

Format each course title as a clickable link, for example:
# [Course Title](course_link)"""
    
    return ChatPromptTemplate.from_template(template)

def initialize_rag_chain():
    """Initialize the RAG pipeline with improved retrieval"""
    try:
        # Load documents and links
        docs = load_course_data()
        links = load_course_links()
        
        # Add links to document metadata
        for i, doc in enumerate(docs):
            doc.metadata["links"] = links
        
        # Modified text splitter with larger chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\nTitle:", "\n\nTitle:", "Title:", ";"]
        )
        splits = text_splitter.split_documents(docs)
        
        # Add links to split documents' metadata
        for split in splits:
            split.metadata["links"] = links
        
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['openai_api_key'])
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chromadb"
        )
        
        # Enhanced retriever configuration
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2,
            max_tokens=4000,
            openai_api_key=st.session_state['openai_api_key']
        )
        
        # Create and return RAG chain
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | get_prompt_template()
            | llm
            | StrOutputParser()
        )
    
    except Exception as e:
        st.error(f"Error initializing RAG chain: {str(e)}")
        return None

def create_sidebar():
    """Create the sidebar with information and tips"""
    with st.sidebar:
        st.header("About")
        st.write("""This chatbot helps you discover courses based on your interests and needs. 
        It provides detailed information about:
        - Course content and curriculum
        - Key highlights
        - Why it's relevant to your interests""")
        
        st.header("Tips")
        st.write("""- Be specific about your interests
        - Ask about particular topics or skills
        - Compare different courses
        - Ask for recommendations based on your level""")
        
        # Add reset button
        if st.button("Reset Chat"):
            st.session_state['messages'] = []
            st.session_state['rag_chain'] = initialize_rag_chain()
            st.rerun()

def main():
    """Main function to run the Streamlit app"""
    # Load environment variables
    load_dotenv()
    
    # Set page config
    st.set_page_config(
        page_title="Course Recommendation Chatbot",
        page_icon="🎓",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Create main UI
    st.title("🎓 Course Recommendation Chatbot")
    st.write("Ask me about our courses! I'll help you find the perfect learning path.")
    
    # Initialize RAG chain if not already initialized
    if st.session_state['rag_chain'] is None:
        with st.spinner("Initializing the course advisor..."):
            st.session_state['rag_chain'] = initialize_rag_chain()
    
    # Create sidebar
    create_sidebar()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("What courses are you interested in?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant courses..."):
                if st.session_state['rag_chain'] is not None:
                    try:
                        response = st.session_state['rag_chain'].invoke(query)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                else:
                    st.error("System not properly initialized. Please try refreshing the page.")

if __name__ == "__main__":
    main()
