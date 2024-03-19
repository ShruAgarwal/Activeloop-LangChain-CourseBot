from langchain.vectorstores import DeepLake
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]

# ------ Data Retrieval Process ------
@st.cache_resource()
def data_lake():
    embeddings = CohereEmbeddings(model = "embed-english-v2.0")

    dbs = DeepLake(
        dataset_path="hub://shruAg01/educhain_course_chatbot", 
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
    memory=ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True, 
        output_key='answer'
        )
    return memory

memory=memory()


# ---------- Initiates the LLM Chat model ----------
model = ChatOpenAI(temperature=0,
        model='gpt-3.5-turbo',
        streaming=True, verbose=True,
        temperature=0,
        max_tokens=1500)


# ---------- Builds the Conversational Chain ----------
qa = ConversationalRetrievalChain.from_llm(
llm=llm,
retriever=compression_retriever,
memory=memory,
verbose=True,
chain_type="stuff",
return_source_documents=True
)

# ---------------------- ToDo: Chat UI -------------------
