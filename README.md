# Course Advisor AI 🎓

An intelligent course recommendation system powered by RAG (Retrieval Augmented Generation) that helps users find and explore free AI/ML courses from Analytics Vidhya's platform.

Link- https://ragbasedsearchtool2-nxf92n39ayb7kmjsychgzh.streamlit.app/

## 🌟 Features

- **Smart Course Recommendations**: Uses advanced RAG architecture to provide contextual course suggestions
- **Interactive Chat Interface**: Clean and intuitive Streamlit-based chat interface
- **Real-time Course Analysis**: Analyzes course content, curriculum, and descriptions to match user queries
- **Free Course Focus**: Specifically designed to help users find free educational resources
- **Dark Mode UI**: Modern, dark-themed interface for better readability
- **Intelligent Retrieval**: Uses FAISS for efficient similarity search and retrieval
- **Structured Response Format**: Provides organized information about course titles, descriptions, and curriculum

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Language Models**
- **Vector Store**: FAISS
- **Embeddings**: OpenAI Embeddings
- **Web Scraping**: Selenium
- **Framework**: LangChain
- **Data Processing**: Python

## 📋 Prerequisites

- Python 3.9+
- LLM API key
- Required Python packages (see requirements.txt)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/course-advisor-ai.git
cd course-advisor-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## 📁 Project Structure

```
course-advisor-ai/
├── app.py                 # Main Streamlit application
├── scraper.py            # Web scraping script for course data
├── course_details.txt    # Scraped course information
├── course_links.txt      # Course URLs from Analytics Vidhya
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables
└── README.md            # Project documentation
```

## 🏃‍♂️ Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

## 🤖 How It Works

1. **Data Collection**: 
   - Uses Selenium to scrape course information from Analytics Vidhya
   - Extracts course titles, descriptions, and curriculum details

2. **RAG Implementation**:
   - Processes course data into embeddings
   - Uses FAISS for efficient similarity search
   - Retrieves relevant course information based on user queries

3. **User Interaction**:
   - Accepts natural language queries
   - Processes queries through the RAG pipeline
   - Returns structured, relevant course recommendations

## 💡 Sample Queries

- "What courses cover RAG systems?"
- "Tell me about LLM selection courses"
- "What are the key topics in the RAG course?"
- "Compare the available courses"
- "Which course is best for beginners?"

## 🎯 Key Components

- **CourseAdvisor Class**: Main application class handling all core functionality
- **RAG Chain**: Implements the retrieval-augmented generation pipeline
- **Custom UI**: Enhanced Streamlit interface with dark mode and modern styling
- **Error Handling**: Robust error handling and logging throughout the application

## 🤝 Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Analytics Vidhya for providing free courses
- Streamlit for the web framework
- LangChain for the RAG implementation framework

---
Created by Arihant Saxena
if you wanna connect - thearihantsaxena@gmail.com
