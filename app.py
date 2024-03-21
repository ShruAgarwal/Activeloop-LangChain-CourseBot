#from langchain.vectorstores import DeepLake
#from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import DeepLake
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
import io
import re
import sys
from typing import Any, Callable
from data_loading import run_job
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
activeloop_org_id = st.secrets["ACTIVELOOP_ORG_ID"]
activeloop_db_id = st.secrets["ACTIVELOOP_DB_ID"]

# Task that will run only once to gather the data needed
# for providing context to the chatbot
run_job()

# ===============================
# PAGE CONFIG
# st.set_page_config(
#     page_title="Activeloop EduChain Bot",
#     page_icon="🦜")


# ------ Data Retrieval Process ------
@st.cache_resource()
def data_lake():
    embeddings = CohereEmbeddings(model = "embed-english-v2.0")

    dbs = DeepLake(
        dataset_path=f"hub://{activeloop_org_id}/{activeloop_db_id}", 
        read_only=True, 
        embedding_function=embeddings
        )
    retriever = dbs.as_retriever()

    # DeepLake instance as a retriever to fetch specific params 
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    # -- Refines and Ranks documents in alignment with a user’s search criteria --
    # This endpoint acts as the last stage reranker of a search flow.
    compressor = CohereRerank(
        model = 'rerank-english-v2.0',
        top_n=5
        )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    return dbs, compression_retriever, retriever

dbs, compression_retriever, retriever = data_lake()


# ---------- Setting up a Memory System for the ChatBot ----------
@st.cache_resource()
def memory():
    # only keeps list of last K interactions
    memory=ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True, 
        output_key='answer'
        )
    return memory

memory=memory()


# ---------- Initiates the LLM Chat model ----------
llm = ChatOpenAI(temperature=0,
        model='gpt-3.5-turbo',
        streaming=True,
        max_tokens=1000)


# ---------- Builds the Conversational Chain ----------
qa = ConversationalRetrievalChain.from_llm(
llm=llm,
retriever=compression_retriever,
memory=memory,
verbose=True,
chain_type="stuff",
return_source_documents=True
)


# ---------------------- Chat UI -------------------
# APP TITLE
c1, c2 = st.columns([0.6, 5], gap="large")

with c1:
    st.image(
        'icon.png',
        width=96,
    )

with c2:
    st.title("Chat with EduChain Bot")
    st.markdown("*AI-Powered LangChain Course Companion 🤖*")

# =========================
# APP INFO EXPANDER
with st.expander("ABOUT THE CHATBOT 👀"):
    st.image('course_banner.png', width=650)
    st.markdown('Explore the Course **[here](https://learn.activeloop.ai/courses/langchain)**')
    st.write('Check the Github repo for detailed info! :)')

# =========================
# Triggers the clearing of the cache and session states
if st.sidebar.button("Start a New Chat Interaction"):
    clear_cache_and_session()

# Initializes chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------- Verbose Display Code --------

def capture_and_display_output(func: Callable[..., Any], args, **kwargs) -> Any:
    # Capture the standard output
    original_stdout = sys.stdout
    sys.stdout = output_catcher = io.StringIO()

    # Run the given function and capture its output
    response = func(args, **kwargs)

    # Reset the standard output to its original value
    sys.stdout = original_stdout

    # Clean the captured output
    output_text = output_catcher.getvalue()
    clean_text = re.sub(r"\x1b[.?[@-~]", "", output_text)

    # Custom CSS for the response box
    st.markdown("""
    <style>
        .response-value {
            border: 2px solid #6c757d;
            border-radius: 5px;
            padding: 20px;
            background-color: #f8f9fa;
            color: #3d3d3d;
            font-size: 20px;  # Change this value to adjust the text size
            font-family: monospace;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("See Langchain Thought Process"):
        # Display the cleaned text as code
        st.code(clean_text)

    return response


# ------ Function for handling chat interactions ------
def chat_ui(qa):
    # Accept user input
    if prompt := st.chat_input(
        "Ask me questions: How can I retrieve data from Deep Lake in Langchain?"
    ):

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Load the memory variables, which include the chat history
            memory_variables = memory.load_memory_variables({})

            # Predict the AI's response in the conversation
            with st.spinner("Searching course material"):
                response = capture_and_display_output(
                    qa, ({"question": prompt, "chat_history": memory_variables})
                )

            # Display chat response
            full_response += response["answer"]
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # Display top 2 retrieved sources
            source = response["source_documents"][0].metadata
            source2 = response["source_documents"][1].metadata
            with st.expander("See Resources"):
                st.write(f"Title: {source['title'].split('·')[0].strip()}")
                st.write(f"Source: {source['source']}")
                st.write(f"Relevance to Query: {source['relevance_score'] * 100}%")
                st.write(f"Title: {source2['title'].split('·')[0].strip()}")
                st.write(f"Source: {source2['source']}")
                st.write(f"Relevance to Query: {source2['relevance_score'] * 100}%")

        # Append message to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# Run function passing the ConversationalRetrievalChain
chat_ui(qa)
