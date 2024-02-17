from transformers import BertTokenizer, BertModel
from docx import Document
import pandas as pd
import torch
from pinecone import Pinecone

def tokenize_chunk_embed(text):
    """
    Create tokens from input text, chunk and embed the tokens.
    Parameters:
        text (str): paragraph text
    Returns: 
        chunked_sequences (list): list of chunked tokens
        embeddings (list): list of embeddings for each chunk
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    max_chunk_size = 512  # chunk size
    # Tokenize each sequence and chunk the tokens
    chunked_sequences = []
    embeddings = []
    tokens_dict = tokenizer(text, return_tensors='pt')
    input_ids = tokens_dict['input_ids']
    # Chunk the tokens
    for i in range(0, input_ids.size(1), max_chunk_size-20):
      chunk = input_ids[:, i:i + max_chunk_size]
      chunked_sequences.append(chunk)
    # Process each chunk
    for chunk in chunked_sequences:
      # Forward pass through BERT model
      with torch.no_grad():
        chunk_outputs = model(input_ids=chunk)
        last_hidden_states = chunk_outputs.last_hidden_state
        single_embedding = torch.mean(last_hidden_states, dim=1).view(-1)  #take average of to get correct shape of embedding
        embeddings.append(single_embedding)
    return chunked_sequences, embeddings

def grab_data():
    """
    Grab text data and create embeddings and chunks for paragraph text.

    Returns: 
        df_chunk_tokens (DataFrame): list of chunked tokens
    """
    allChunks = []
    allEmbeddings = []
    doc = Document('blackstudies.docx')  #return document object with data
    for paragraph in doc.paragraphs:
        fileChunks, fileEmbeddings = tokenize_chunk_embed(paragraph.text)
        allChunks.extend(fileChunks)
        allEmbeddings.extend(fileEmbeddings)

    allChunks = [x[0] for x in allChunks]
    df_chunk_tokens = pd.DataFrame({'chunks': allChunks,  'embeddings': allEmbeddings})
    return df_chunk_tokens

def store_pinecone(df):
    """
    Store the embeddings and respective texts in Pinecone index.

    Parameters:
        df(DataFrame): dataframe containing embeddings and associated text.
    """
    pc = Pinecone(api_key="17944aa6-d5f5-48f6-843a-aa1e79415857")
    index = pc.Index("590-llm-project")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, row in df.iterrows():
        token_chunk = row['chunks']
        embedding = row['embeddings'].numpy().tolist()
        embedding_dict = {}
        embeddingList = []
        embedding_dict['id'] = str(i)         # unique id for each embedding
        embedding_dict['values'] = embedding
        embedding_dict['metadata'] = {}       #metadata for Pinecone must be in dictionary format
        embedding_dict['metadata']['chunk'] = tokenizer.decode(token_chunk)   #in metadata dictionary, create chunk key with value text
        embeddingList.append(embedding_dict)
        index.upsert(vectors=embeddingList)
    print("Pinecone storage complete.")

def main():
   df_chunk_tokens = grab_data()
   store_pinecone(df_chunk_tokens)

if __name__ == "__main__":
   main()