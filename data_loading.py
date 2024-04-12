from langchain_community.utilities import ApifyWrapper
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import DeepLake
import streamlit as st
import os
os.environ["APIFY_API_TOKEN"] = st.secrets["APIFY_API_TOKEN"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]
activeloop_org_id = st.secrets["ACTIVELOOP_ORG_ID"]


@st.cache_resource()
def run_job():
    # ------ Scrapes the content from websites ------
    apify = ApifyWrapper()
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": "https://learn.activeloop.ai/courses/langchain"}]},  # course URL to scrape
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

    username = activeloop_org_id                # get yours from app.activeloop.ai
    db_id = "activeloop_course_educhain_bot"       # replace with your Database name
    #DeepLake.force_delete_by_path(f"hub://{username}/{db_id}")


    # ---- Stores and retrieves the transformed data ----
    dbs = DeepLake(dataset_path=f"hub://{username}/{db_id}", embedding=embeddings)
    dbs.add_documents(docs_split)