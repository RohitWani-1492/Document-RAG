# Document-RAG

# 📄 PDF Chatbot with LangChain & Streamlit

A simple chatbot application built with **LangChain**, **OpenAI**, and **Streamlit** that allows you to **upload a PDF** and interact with its contents by asking questions in natural language.

---

## 🚀 Features
- Upload any PDF file via the sidebar.
- Extracts and processes the text automatically.
- Splits large documents into smaller chunks for efficient retrieval.
- Uses **FAISS** as a vector database for similarity search.
- Retrieves the most relevant sections from the PDF to answer questions.
- Powered by **OpenAI GPT models** for conversational responses.
- Clean chat-style interface using **Streamlit**.

---

## 🛠️ Tech Stack
- **Python**
- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – Document processing & chaining
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search
- [OpenAI](https://platform.openai.com/) – Embeddings & LLM
- [dotenv](https://pypi.org/project/python-dotenv/) – Environment variable management

---

## 📂 Project Structure
├── document_rag.py # Main Streamlit app
├── requirements.txt # Dependencies
├── .env # API key (not committed to GitHub)
└── README.md # Project documentation


---

## ⚙️ Installation & Setup

1. **Clone this repository**
   git clone https://github.com/RohitWani-1492/DOCUMENT-RAG.git
   cd DOCUMENT-RAG
   
2. **Create a virtual environment**
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   
3. **Install dependencies**
   pip install -r requirements.txt

4. **Add your OpenAI API key**
   Create a .env file in the project root.
   Add the following line:
      OPENAI_API_KEY=your_api_key_here

5. **Run the Streamlit app**
    streamlit run document_rag.py

6. **Open in browser**
    The app will run at: http://localhost:8501

## 📖 How It Works

1. Upload a PDF file.
2. The app extracts text, splits it into manageable chunks, and stores embeddings in FAISS.
3. When you ask a question:
4. Relevant chunks are retrieved using similarity search.
5. Context + your question are passed to the LLM.
6.The chatbot generates a contextual answer.

## Demo Screenshot

<img width="1917" height="1079" alt="Screenshot 2025-09-01 183855" src="https://github.com/user-attachments/assets/88aa0cb8-d69c-434f-ae30-86d6a97178bf" />

## 🔮 Future Improvements

1. Support multiple file uploads.
2. Add support for other file types (Word, TXT).
3. Enable long-term memory across sessions.
4. Deploy on Streamlit Cloud or Hugging Face Spaces.
