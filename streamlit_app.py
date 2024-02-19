
import streamlit as st
import json
import pandas as pd
import subprocess
from pinecone import Pinecone
import os
import base64
from openai import OpenAI
import streamlit_scrollable_textbox as stx
from rag import grab_relevant_information, generate_response, generate_response_noRAG

def load_page_components():
    st.set_page_config(page_title="Home", layout="wide")
    background_html = f'''
        <style>
            body {{
                margin: 0;
                padding: 0;
            }}

            .image-container {{
                display: flex;
                position: fixed;
                height: 100vh;
                width: 100vw;
                right: 0;
                bottom: 0;
            }}
            .full-image {{
                background-image: url('data:image/jpg;base64,{base64.b64encode(open('background.jpg', 'rb').read()).decode()}');
                width: 100vw; 
                background-size: cover;
                background-position: center;
                background-position: 30% 30%;
                height: 100vh;
                opacity: 0.4;
            }}
        </style>
        <div class="image-container">
            <div class="full-image"></div>
        </div>
    '''
    # Render the background using st.markdown with unsafe_allow_html=True
    st.markdown(background_html, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white;'>The History of Black Studies at Duke</h1>", unsafe_allow_html=True)

    user_input = st.text_input("ask away...")
    button = st.button("Generate Response")
    return user_input, button

def create_datatable(outputFile):
    with open("./output.txt", 'r') as file:   #replace with outputFile
        vector_ids = file.read()

    id_List = json.loads(vector_ids)['vectors']
    ids = [x['id'] for x in id_List]

    pc = Pinecone(api_key="17944aa6-d5f5-48f6-843a-aa1e79415857")
    index = pc.Index("590-llm-project")
    response = index.fetch(ids = ids)

    embeddings = [response['vectors'][x]['values'] for x in ids]
    chunks = [response['vectors'][x]['metadata']['chunk'] for x in ids]
    df_chunk_tokens = pd.DataFrame({'chunks': chunks,  'embeddings': embeddings})
    return df_chunk_tokens

def grab_data_ids(outputFilePath):
    PINECONE_API_KEY="17944aa6-d5f5-48f6-843a-aa1e79415857"
    INDEX_HOST="590-llm-project-rgw9xaj.svc.apw5-4e34-81fa.pinecone.io"

    command = ["curl", "X", f"https://{INDEX_HOST}/vectors/list?", "-H", f"Api-Key: {PINECONE_API_KEY}"]

    with open(outputFilePath, 'w') as file:
        subprocess.run(command, check = True, stdout=file)

def populate_columns(RAG_response, no_RAG_response):
    col1, col2 =  st.columns(2)
    with col1:
        st.header("no-RAG Response")
        stx.scrollableTextbox(no_RAG_response, 300, border = False)
    with col2:
        st.header("RAG Response")
        stx.scrollableTextbox(RAG_response, 300, border = False)


def main(local_dir = os.getcwd()):
    user_input, button = load_page_components()
    output_path = os.path.join(local_dir, "output.txt")
    grab_data_ids(output_path)
    df_chunks_embeddings = create_datatable(output_path)
    if button:
        top_info = grab_relevant_information(user_input, df_chunks_embeddings)
        client = OpenAI()
        RAG_response = generate_response(client, user_input, top_info)
        no_RAG_response = generate_response_noRAG(client, user_input)
        populate_columns(RAG_response, no_RAG_response)
        
if __name__ == "__main__":
    main()




