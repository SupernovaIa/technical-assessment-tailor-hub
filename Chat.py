# Web application framework
# -----------------------------------------------------------------------
import streamlit as st

# Environment and system utilities
# -----------------------------------------------------------------------
import os
import dotenv

# LangChain models and schemas
# -----------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Custom modules
# -----------------------------------------------------------------------
from src.rag import load_doc_to_db, stream_llm_rag_response, stream_llm_response

# Load dot env
dotenv.load_dotenv()

# Streamlit Application Setup
# -----------------------------------------------------------------------

# Streamlit page config
st.set_page_config(
    page_title="Video Chatbot", 
    page_icon="🎬", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# API Key
with st.sidebar:

    default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
    
    with st.popover("OpenAI API Key", icon="🔒"):
        openai_api_key = st.text_input(
            "Introduce your OpenAI API Key", 
            value=default_openai_api_key, 
            type="password",
            key="openai_api_key"
        )

# Checking if the user has introduced the OpenAI API Key
missing_openai = openai_api_key is None

if missing_openai:
    st.warning("⬅️ Please introduce an API Key to continue!")

else:
    # Sidebar
    with st.sidebar:

        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="secondary")

        temp = st.slider("Temperatue", min_value=0.0, max_value=1.0, value=0.3, step=0.1, help="Select model temperature.")

    # Main chat app
    llm_stream = ChatOpenAI(
        api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=temp,
        streaming=True,
    )

    # Message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Prompt and chat
    if "vector_db" in st.session_state:
        if prompt := st.chat_input("Type here your message"):
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
                st.write_stream(stream_llm_rag_response(llm_stream, messages))

    # Ensure session_id is defined no
    else:
        st.session_state.rag_sources = []
        st.title("Please, load RAG docs to start chatting")
        # Load RAG documents
        is_vector_db_loaded = False
        st.button("Load RAG docs", on_click=load_doc_to_db, type="primary", disabled = is_vector_db_loaded, help="This only works if any transcription available. Go to Load videos to get some.")
