# RAG -> Retrieval Augmented Generation
# LLM -> Large Language Model
# LLMs are good at generating text but may not have up-to-date or specific information
# RAG combines LLMs with a retrieval system to provide more accurate and relevant responses

# Load the necessary packages
import streamlit as st
import os

# langchain_cohere is a package that provides integration with Cohere's language models
# CohereEmbeddings is a class that allows you to generate embeddings using Cohere's models
from langchain_cohere import CohereEmbeddings

# langchain_groq is a package that provides integration with Groq's language models
# ChatGroq is a class that allows you to interact with Groq's chat-based language models
from langchain_groq import ChatGroq

# langchain_community is a package that provides community-contributed integrations and tools for LangChain
# document_loaders is a module that contains various document loaders for different file formats
# PyPDFLoader is a class that allows you to load and parse PDF documents
from langchain_community.document_loaders import PyPDFLoader

# langchain_text_splitters is a package that provides various text splitting strategies
# RecursiveCharacterTextSplitter is a class that splits text into smaller chunks based on character count
from langchain_text_splitters import RecursiveCharacterTextSplitter

# langchain_chroma is a package that provides integration with Chroma, a vector database
# Chroma is a class that allows you to create and manage a Chroma vector store
from langchain_chroma import Chroma

# langchain_core is the core package of LangChain that provides essential components and utilities
# prompts is a module that contains various prompt templates and utilities
# ChatPromptTemplate is a class that allows you to create chat-based prompt templates
from langchain_core.prompts import ChatPromptTemplate

# langchain_chains is a package that provides various chain implementations for building complex workflows
# combine_documents is a module that contains chains for combining multiple documents into a single response
# create_stuff_documents_chain is a function that creates a chain for "stuffing" documents into a prompt
from langchain.chains.combine_documents import create_stuff_documents_chain

# create_retrieval_chain is a function that creates a retrieval-augmented generation chain
from langchain.chains import create_retrieval_chain

# Set the API keys for Cohere and Groq from environment variables
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# To create a temporary directory to store uploaded files (PDF's)
# If the directory already exists, it won't raise an error
# makedirs is used to create directories recursively
os.makedirs("temp", exist_ok=True)

# Load the Groq chat and Cohere model
# CohereEmbeddings is used to create embeddings for text data
# ChatGroq is used to interact with Groq's chat-based language models
# temperature controls the randomness of the model's output
# max_tokens sets the maximum number of tokens the model can generate in a single response
embedding_model = CohereEmbeddings(model="embed-english-v3.0")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, max_tokens=500)

# Helper function to process the uploaded PDF file
# Load and split the document into smaller chunks
def load_pdf_and_split (path: str):
    # Load the PDF document using PyPDFLoader
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Split the document into smaller chunks using RecursiveCharacterTextSplitter
    # chunk_size is the maximum size of each chunk
    # chunk_overlap is the number of overlapping characters between chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# cache_resource is used to cache the vector store
# This avoids reloading and reprocessing the documents every time the app is run
@st.cache_resource(show_spinner="Loading the vector store...")

# Create and store the vector store
def get_vectorstore (splits, persist_dir):
    # Create a Chroma vector store from the document chunks
    # splits are the document chunks to be stored
    # persist_dir is the directory where the vector store will be persisted
    # embedding_model is used to generate embeddings for the document chunks
   return Chroma.from_documents(
        splits, embedding=embedding_model, persist_directory=persist_dir
    )

# Get the response from the Model
def get_response(retriever, query: str):
    # RAG pipeline using stuff chain with long-answer friendly prompt
    # Define the system prompt with instructions for the model
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer the question "
        "as completely as possible. If the answer is not in the context, say that you don't know, "
        "but always explain your reasoning based on the available information. "
        "Provide detailed answers and examples when relevant.\n\n{context}"
    )
    # Create a chat prompt template with system and human messages
    # system message provides instructions to the model
    # human message contains the user's query
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # Create a question-answering chain using the prompt and the LLM
    # create_stuff_documents_chain is used to create a chain that stuffs documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create a retrieval-augmented generation chain using the retriever and the question-answering chain
    # create_retrieval_chain combines the retriever and the question-answering chain to provide
    # relevant context to the LLM for generating responses
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Invoke the RAG chain with the user's query and get the response
    response = rag_chain.invoke({"input": query})
    return response["answer"]

# List all persisted PDFs
def list_documents():
    # Return list of all persisted PDFs
    return [
        # Remove the 'chroma_db_' prefix from folder names to get document names
        folder.replace("chroma_db_", "")
        for folder in os.listdir(".")
        # Only include folders that start with 'chroma_db_'
        if folder.startswith("chroma_db_")
    ]

# Streamlit app layout
st.set_page_config(page_title="RAG with Groq and Cohere", page_icon=":books:")
st.title("📚 RAG with Groq and Cohere")

# Add db and selected document into the state
# This allows us to persist the vector store and selected document across user interactions
# db will hold the Chroma vector store
# selected_document will hold the name of the currently selected document
if "db" not in st.session_state:
    st.session_state.db = None
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# Creating a Query for the User to Input
query = st.text_input("Please enter your question here:")

# Button to Submit the Query
submit = st.button("Submit")

# Creating a Sidebar for Uploading and Selecting Documents
with st.sidebar:
    st.header("Upload and Select Document")
    # File uploader to upload PDF documents
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    # If a file is uploaded, process it
    if uploaded_file:
        temp_file = os.path.join("temp", uploaded_file.name)
        # Save the uploaded file to the 'temp' directory
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Button to Create Embeddings and Store in Vector DB
        if st.button("Please click to create embeddings and store in vector DB"):
            with st.spinner("Processing the document..."):
                # Load and split the PDF document into chunks
                splits = load_pdf_and_split(temp_file)

                # Create and store the vector store in a directory named 'chroma_db_<filename>'
                persist_dir = f"./chroma_db_{uploaded_file.name}"
                st.session_state.db = get_vectorstore(splits, persist_dir=persist_dir)
                st.session_state.selected_doc = uploaded_file.name

                ############ Feedback to User ############
                # Notify the user that the embeddings have been created and stored
                # .success displays a success message in Streamlit
                st.success(f"Embeddings for '{uploaded_file.name}' uploaded and processed successfully!")

    # List all persisted documents for selection
    docs = list_documents()
    if docs:
        selected_doc = st.selectbox("Select Document", docs,
            # index the selected document if it exists in the list, otherwise default to the first document
            index=(
                docs.index(st.session_state.selected_doc)
                if st.session_state.selected_doc in docs
                # 0 means the first document in the list
                else 0
            ),
        )
        st.session_state.selected_doc = selected_doc
        st.session_state.db = Chroma(
            # Load the vector store from the directory corresponding to the selected document
            # f is used to format the string with the selected document name
            persist_directory=f"./chroma_db_{selected_doc}",
            # embedding_function is used to generate embeddings for the document chunks
            embedding_function=embedding_model,
        )
    else:
        st.info("No documents found. Please upload a PDF document.")

# Handle the query submission
if submit:
    if not query:
        st.warning("Please enter a question.")
    elif not st.session_state.db:
        st.warning("Please upload a document and create embeddings first.")
    else:
        with st.spinner("Generating response..."):
            # Get the retriever from the vector store
            retriever = st.session_state.db.as_retriever()
            # Get the response from the model using the retriever and the user's query
            answer = get_response(retriever, query)
            # Display the answer to the user
            st.markdown("Answer:")
            st.write(answer)