# Environment configuration
# -----------------------------------------------------------------------
import os
import dotenv

# Web application framework
# -----------------------------------------------------------------------
import streamlit as st

# LangChain utilities for document processing and retrieval
# -----------------------------------------------------------------------
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


dotenv.load_dotenv()

DOCS_DIR = "transcriptions/"

# LLM response streaming
def stream_llm_response(llm_stream, messages):
    """
    Streams responses from an LLM and updates the session state.

    Parameters
    -----------
    - llm_stream (object): An object capable of streaming responses from the LLM.
    - messages (list): A list of message dictionaries representing the conversation history.

    Yields
    -------
    - chunk (object): A chunk of the streamed response.

    Side Effects
    ------------
    - Updates `st.session_state.messages` by appending the complete response.
    """
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# Vector database and docs vectorization
def load_doc_to_db():
    """
    Loads text documents from a specified directory into the database.

    Checks if the directory exists, filters supported document types, 
    loads the content, and processes it for storage.
    """
    if not os.path.exists(DOCS_DIR):
        st.error(f"❌ Directory '{DOCS_DIR}' does not exist.")
        return

    docs = []
    supported_types = {"txt", "md"}
    
    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        
        if os.path.isfile(file_path) and filename.split(".")[-1] in supported_types:
            try:
                loader = TextLoader(file_path)
                docs.extend(loader.load())
                st.write(f"📄'{filename}' document successfully loaded.")
            except Exception as e:
                st.warning(f"⚠ Error loading '{filename}': {e}")

    if docs:
        try:
            _split_and_load_docs(docs)
            st.success(f"✅ {len(docs)} documents have been loaded in the database.")
        except Exception as e:
            st.error(f"❌ Failed to process documents: {e}")


def initialize_vector_db(docs):
    """
    Initializes a vector database from a list of documents.

    Parameters
    -----------
    - docs (list): A list of document objects to be embedded and stored.

    Returns
    --------
    - vector_db (Chroma or None): A Chroma vector database instance if initialization is successful.
    """
    embedding = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding
    )

    return vector_db


def _split_and_load_docs(docs):
    """
    Splits documents into smaller chunks and loads them into a vector database.

    Parameters
    -----------
    - docs (list): A list of document objects to be split and stored.

    Side Effects
    ------------
    - Uses `RecursiveCharacterTextSplitter` to split documents into chunks.
    - Initializes or updates the vector database stored in `st.session_state.vector_db`.
    - Displays status messages using Streamlit (`st.success`, `st.error`).

    Raises
    ------
    - Displays an error if the vector database initialization fails.
    - Displays an error if adding document chunks to the database fails.
    """
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state or st.session_state.vector_db is None:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
        
        if st.session_state.vector_db:
            st.success("✅ Vector database initialized with document chunks.")
        else:
            st.error("❌ Failed to initialize the vector database.")
    else:
        try:
            st.session_state.vector_db.add_documents(document_chunks)
            st.success(f"✅ {len(document_chunks)} new document chunks added.")
        except Exception as e:
            st.error(f"❌ Error adding documents: {e}")


def _get_context_retriever_chain(vector_db, llm):
    """
    Creates a context-aware retriever chain for retrieving relevant information from a vector database.

    Parameters
    -----------
    - vector_db (Chroma): A Chroma vector database instance used for retrieving documents.
    - llm (object): A language model instance used to generate search queries.

    Returns
    --------
    - retriever_chain (object): A history-aware retriever chain for contextual information retrieval.
    """
    retriever = vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages."),
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(llm):
    """
    Creates a Conversational RAG (Retrieval-Augmented Generation) chain for answering queries related to the 2024 Australian Grand Prix.

    Parameters
    -----------
    - llm (object): A language model instance used for generating responses.

    Returns
    --------
    - rag_chain (object): A retrieval chain that retrieves relevant context and generates responses.
    """
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are an AI assistant designed to answer questions about YouTube and Vimeo videos.
        Your goal is to provide accurate, detailed, and relevant responses based on the video's content.
        If you are asked about a video use retrieved context for answering. 
        You will always have context for answering so use it, never say you don't have the information.
        {context}"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    """
    Streams a response from the Conversational RAG chain and updates the session state.

    Parameters
    -----------
    - llm_stream (object): An object capable of streaming responses from the LLM.
    - messages (list): A list of message dictionaries representing the conversation history.

    Yields
    -------
    - chunk (str): A streamed chunk of the response text.
    """
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"

    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})