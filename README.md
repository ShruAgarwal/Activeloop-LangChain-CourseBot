# 🤖 Chat with EduChain Bot
### An LLM companion for answering your questions related to the [LangChain & Vector DBs in Production course](https://learn.activeloop.ai/courses/langchain) brought by *[Activeloop](https://www.activeloop.ai/)*

## Demo 🕹
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://educhain-bot.streamlit.app/)

https://github.com/ShruAgarwal/Activeloop-LangChain-CourseBot/assets/82811717/55c29e11-cc27-4e9d-8d69-848a48e724e8

## How to use 👀

1. Enter your OpenAI API key.
   - You can get your own OpenAI API key from [here](https://platform.openai.com/account/api-keys) and then click on the `+ Create new secret key` button.
2. You can now proceed to ask questions related to the course to the chatbot.

## Behind the Scenes ⚙
This educational chatbot demonstrates the power of **Retrieval Augmented Generation (RAG)** to answer queries related to the course and provides relevant info to you by retrieving data from an extensive and detailed knowledge base. It returns a natural response to your questions along with the truth source.

*Here's a summary of the scripts used for building this chatbot:*

1. `data_loading.py`:
   - Handles the initial data gathering and processing task where it scrapes the text data from the Langchain course website using **[Apify](https://apify.com/)**.
   - The scraped text data is then converted into numerical form (vectors) using [`CohereEmbeddings`](https://docs.cohere.com/docs/embeddings) that the chatbot can learn from.
   - Finally, the transformed data is uploaded to **[Deep Lake](https://docs.activeloop.ai/)**, a data storage service, for future use.

2. `app_workflow.py`:
   - Handles the retrieval and ranking of the relevant data.
   - First, it gathers the stored data from the `data_loading.py` file.
   - Second, it uses `CohereRerank` to rank and retrieve the most relevant data based on the user’s query.
      - `CohereRerank` is a reranking service that refines and ranks documents in alignment with a user’s search criteria.
   - Third, it also builds the conversation chain with memory, which helps in maintaining the context of the conversation.

3. `app_demo.py`:
   - The main script demonstrates the working of the chatbot through a user-friendly web interface using **Streamlit**.
   - The chatbot then uses the stored and transformed data from **Deep Lake** to answer user queries.

### Tech-stack 🛠
<p align="center">
  <img src="https://github.com/ShruAgarwal/Activeloop-LangChain-CourseBot/blob/main/tech_stack.png"/>
</p>

## Key Learnings 🌱
- Provided a deep understanding of how RAG can be used to answer queries by retrieving relevant information from a detailed knowledge base.
- Involves scraping data from a course website, which helped in understanding how to extract and structure data from the web.
- The use of **Cohere** for embedding and reranking provided insights into how these techniques can improve the relevance of the retrieved information.
- Storing the transformed data in Deep Lake helped in understanding the importance of efficient data storage and retrieval in AI applications.
- Demonstrated how to integrate powerful APIs and libraries like **OpenAI and Langchain** to build a sophisticated chatbot.
- Highlightes the potential of AI in enhancing educational experiences, by providing a chatbot that can answer course-related queries.
- Helped with learning *project management, problem-solving, and debugging skills* to bring all the components together into a working chatbot.

## Credits ✨
- *Inspired to build this chatbot from* [this tutorial!](https://www.activeloop.ai/resources/retrieval-augmented-generation-for-llm-bots-with-lang-chain/)
- *Thanks to* [Yuichiro's Streamlit Theme Editor](https://github.com/whitphx/streamlit-theme-editor) that helped me find the suitable app's theme :)
- *Chatbot logo and tech stack design made by me using [Canva](https://www.canva.com/)*
