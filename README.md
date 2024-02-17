# RAG: Duke Black Studies
In this project, I use documents, newspaper articles, letters, and more from the digital Duke Black History Archive of the Rubenstein, along with RAG to allow users to learn more about an important Duke topic during Black History Month!
## Scripts
- extract_store_data.py: grabs data from docx file, tokenizes, chunks, embeds, and stores both the embeddings and the correlated text chunks into Pinecone
- rag.py: includes the functions needed (imported and called in streamlit_app.py) to perform retrieval semantic search, grab the most relevant text chunks, and call OpenAI's ChatGPT to create response to user's question
- streamlit_app.py: run the streamlit frontend

## Notebooks
- extract_embed.ipynb: notebook to extract data and put in Pinecone.
- RAG_and_evaluation: notebook that performs retrieval semantic search, calls ChatGPT using both RAG and nonRAG and performs evaluation on three examples.
## Other Files
- background.jpg: background image for streamlit app (artist credit- Cortney Buford)
- blackstudies.docx: document containing all the necessary texts.

## How to Run
- extract_store_data.py : "python extract_store_data.py" in terminal
    Note: Once extract_store_data.py is run once, you will not need to run again, as data is stored in Pinecone and can be pulled at anytime.
- streamlit_app.py: "streamlit run streamlit_app.py" in terminal

