import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, GlobalAveragePooling1D
from sklearn.metrics.pairwise import cosine_similarity

vocab_size = 30000
glove_dim = 300
sequence_length = 300

reviews_df = pd.read_pickle('reviews_segment.pkl')[['review_id', 'review_text']]

def query(text, tokenizer, model, dataset_embeddings, top_n=5):
    # Tokenize and pad the query
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=sequence_length, padding='post')
    
    # Generate the embedding for the query
    query_embedding = model.predict(padded_seq)
    
    # Compute cosine similarities with dataset embeddings
    similarities = cosine_similarity(query_embedding, dataset_embeddings)[0]
    
    # Find the top N most similar sentences
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_scores = similarities[top_indices]
    print(reviews_df.loc[19391].review_text)
    return top_indices, top_scores

def main():


    # Load the embedding matrix
    embedding_matrix = np.load('embedding_matrix.npy')

    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load dataset embeddings
    dataset_embeddings = np.load('dataset_embeddings.npy')

    # Rebuild the model

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=glove_dim, input_length=sequence_length, 
                        weights=[embedding_matrix], trainable=False))
    model.add(GlobalAveragePooling1D())
    #model.add(Dense(units=128, activation=None))
    model.compile(optimizer='adam', loss='cosine_similarity')  # No training needed

    searchQuery = "battery life very short"
    print(query(searchQuery, tokenizer, model, dataset_embeddings))
if __name__ == "__main__":
    main()