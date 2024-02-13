
import streamlit as st
import json
import pandas as pd
import subprocess
from pinecone import Pinecone
import os
from openai import OpenAI
from rag import grab_relevant_information, generate_response, generate_response_noRAG


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
    #!curl -X GET "https://$INDEX_HOST/vectors/list?"\
    #-H "Api-Key: $PINECONE_API_KEY" > ./output.txt   #replace

    with open(outputFilePath, 'w') as file:
        subprocess.run(command, check = True, stdout=file)

def main(local_dir = os.getcwd()):
    output_path = os.path.join(local_dir, "output.txt")
    grab_data_ids(output_path)
    df_chunks_embeddings = create_datatable(output_path)
    user_input = st.text_input("Ask question please stink!")
    button = st.button("generate response hoe")

    if button:
        top_info = grab_relevant_information(user_input, df_chunks_embeddings)
        os.environ['OPENAI_API_KEY'] = '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'
        client = OpenAI()
        RAG_response = generate_response(client, user_input, top_info)
        no_RAG_response = generate_response_noRAG(client, user_input)
        st.write(f"This is the RAG response {RAG_response}")
        st.write(f"This is the non RAG response {no_RAG_response}")

if __name__ == "__main__":
    main()




