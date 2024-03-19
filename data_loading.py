from langchain.document_loaders import ApifyDatasetLoader
from langchain.utilities import ApifyWrapper
from langchain.document_loaders.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import DeepLake
import streamlit as st
import os
os.environ["APIFY_API_TOKEN"] = st.secrets["APIFY_API_TOKEN"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]


# ------ Scrapes the content from websites ------
apify = ApifyWrapper()
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "ENTER\YOUR\URL\HERE"}]},   # ToDo: Specify the course URL here
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"] if dataset_item["text"] else "No content available",
        metadata={
            "source": dataset_item["url"],
            "title": dataset_item["metadata"]["title"]
        }
    ),
)

docs = loader.load()


# ------ Splits the scraped documents into smaller chunks ------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)
docs_split = text_splitter.split_documents(docs)


# ---- Translates text data into numerical data ----
embeddings = CohereEmbeddings(model = "embed-english-v2.0")

username = st.secrets["ACTIVELOOP_ORG_ID"]   # get yours from app.activeloop.ai
db_id = "educhain_course_chatbot"         # replace with your Database name
DeepLake.force_delete_by_path(f"hub://{username}/{db_id}")

# ---- Stores and retrieves the transformed data ----
dbs = DeepLake(dataset_path=f"hub://{username}/{db_id}", embedding_function=embeddings)
dbs.add_documents(docs_split)