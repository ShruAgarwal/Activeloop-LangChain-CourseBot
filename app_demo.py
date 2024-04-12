from langchain_community.vectorstores import DeepLake
from langchain_cohere import CohereEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere.rerank import CohereRerank
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import streamlit as st
import io
import re
import sys
from typing import Any, Callable
import os
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
activeloop_org_id = st.secrets["ACTIVELOOP_ORG_ID"]
activeloop_db_id = "activeloop_course_educhain_bot"

# ===============================
# PAGE CONFIG
st.set_page_config(
    page_title="EduChain Bot",
    page_icon="🦜")

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
    st.markdown("""**This is an educational chatbot that can answer your questions related
    to the :orange[LangChain & Vector Databases in Production] course by [Activeloop](https://www.activeloop.ai/)**""")
    st.link_button("Explore the Course 📖", "https://learn.activeloop.ai/courses/langchain")
    st.image('course_banner.png', width=650)

# =========================
# ASKING FOR USER'S API KEYS
with st.sidebar:
    st.header('📌 Get Started')
    with st.popover("🔑 Enter Your Keys Here"):
        cohere_api_key = st.text_input('COHERE API KEY:', type='password')
        openai_api_key = st.text_input('OPENAI API KEY:', type='password')
        if not (cohere_api_key and openai_api_key):
            st.warning('Please enter your API keys!', icon='⚠️')
        else:
            st.success('You can now proceed to ask questions to the chatbot! 👉')
    
    st.markdown('---')


# ------ Data Retrieval Process ------
@st.cache_resource()
def data_lake(cohere_api_key):
    embeddings = CohereEmbeddings(model = "embed-english-v2.0", cohere_api_key=cohere_api_key)

    dbs = DeepLake(
        dataset_path=f"hub://{activeloop_org_id}/{activeloop_db_id}", 
        read_only=True, 
        embedding=embeddings
        )
    retriever = dbs.as_retriever(search_type="mmr")

    # DeepLake instance as a retriever to fetch specific params 
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
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

dbs, compression_retriever, retriever = data_lake(cohere_api_key)


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
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=openai_api_key,
    model='gpt-3.5-turbo',
    streaming=True,
    max_tokens=1000
)


# ---------- Builds the Conversational Chain ----------
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_retriever,
    memory=memory,
    verbose=True,
    chain_type="stuff",
    return_source_documents=True
)


# Triggers the clearing of the cache and session states
if st.sidebar.button("Start a New Chat Interaction"):
    clear_cache_and_session()

# -- Part of Chat UI --
st.sidebar.markdown('---')
st.sidebar.write("⚡ *If you don't have the required keys, get them for FREE using the links below!*")
st.sidebar.info("""[COHERE TRIAL API KEY](https://dashboard.cohere.com/api-keys) and
[OPENAI API KEY](https://platform.openai.com/api-keys)""")

st.sidebar.info('⭐ Check out the [Github repo](https://github.com/ShruAgarwal/Activeloop-LangChain-CourseBot) for detailed info!')
st.sidebar.info("*Made by [Shruti Agarwal](https://www.linkedin.com/in/shruti-agarwal-bb7889237)*")

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


def main():

    # Run function passing the ConversationalRetrievalChain
    chat_ui(qa)

if __name__ == "__main__":
    main()