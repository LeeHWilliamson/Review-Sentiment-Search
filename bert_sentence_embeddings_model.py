from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import pickle
import torch.nn.functional as F
from tqdm import tqdm

reviews_df = pd.read_pickle('reviews_segment.pkl')[['review_id', 'review_text']]

def mean_pooling(model_output, attention_mask): #compute embeddings for sentences from tokens
    token_embeddings = model_output[0] #first element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() #expand input mask to take in embeddings of entire sentence
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#text to embed
review_text = reviews_df['review_text'].tolist()
print(f"Number of reviews: {len(review_text)}")
print(f"Example review: {review_text[0]}")

#get tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

batch_size = 32 #sentences are processed in parallel, lets just do 32 at once
sentence_embeddings = []

for i in tqdm(range(0, len(review_text), batch_size), desc = "Encoding Batches"):
    batch_texts = review_text[i:i + batch_size]

    #tokenize reviews
    encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

    #compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #perform pooling (compute sentence embeddings for each batch)
    batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    #normalize sentence embeddings for each batch
    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

#add to main embedding
sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

print("Sentence embeddings: ")
print(sentence_embeddings)