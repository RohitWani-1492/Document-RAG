from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import tempfile

load_dotenv()

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDF")

# Sidebar - upload file
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state for chat history and retriever
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Load and process PDF only once per upload
    if st.session_state.retriever is None:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

    retriever = st.session_state.retriever
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, just say you don't know.
        
        Context:
        {context}
        
        Question: {question}""",
        input_variables=['context', 'question']
    )

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_query := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve and answer
        context = retriever.invoke(user_query)
        context_text = "\n\n".join(doc.page_content for doc in context)
        final_prompt = prompt.invoke({"context": context_text, "question": user_query})
        answer = llm.invoke(final_prompt)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer.content})
        with st.chat_message("assistant"):
            st.markdown(answer.content)

else:
    st.info("📥 Please upload a PDF from the sidebar to start chatting.")
