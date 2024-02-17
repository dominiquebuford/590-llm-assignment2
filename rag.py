
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def grab_relevant_information(user_question, df_chunk_tokens):
  #tokenize and embed input
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  tokens_dict = tokenizer(user_question, return_tensors='pt')
  input_ids = tokens_dict['input_ids']

  with torch.no_grad():
    chunk_outputs = model(input_ids)
    last_hidden_states = chunk_outputs.last_hidden_state

  single_embedding = torch.mean(last_hidden_states, dim=1).numpy()

  embeddings = np.stack(df_chunk_tokens['embeddings'].to_numpy())
  similarityscores= cosine_similarity(single_embedding, embeddings)

  df_chunk_tokens['similarityScore'] = similarityscores[0]

  df_scores_sorted = df_chunk_tokens.sort_values(by = 'similarityScore', ascending = False)

  top_5_responses = ",".join(df_scores_sorted.iloc[:5]['chunks'].tolist())

  return top_5_responses

def generate_response(client, user_question, gathered_text):
  # Create the call to gpt for answering the user's question
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"You are answering the user's question as best you can, according to this information {gathered_text} about Duke's Black history. if you cannot create an accurate answer, tell the user you don't have availability to that information"},
        {"role": "user", "content": f"{user_question}"}
  ]
  )
    return completion.choices[0].message.content

def generate_response_noRAG(client, user_question):
 # Create the call to gpt for answering the user's question
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"You are answering the user's question as best you can about Duke's Black history. if you cannot create an accurate answer, tell the user you don't have availability to that information"},
        {"role": "user", "content": f"{user_question}"}
  ]
  )
    return completion.choices[0].message.content