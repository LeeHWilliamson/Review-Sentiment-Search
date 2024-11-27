#EXAMPLE FROM https://towardsdatascience.com/a-guide-to-word-embeddings-8a23817ab60f
#GOAL IS TO CATEGORIZE QUESTIONS INTO 3 DIFFERENT CATEGORIES BASED ON USEFULNESS
#OUR DATA CONSISTS OF QUESTION TEXT AND CATEGORY ASSIGNMENT FOR EACH QUESTION
#I.E. EACH QUESTION HAS BEEN LABELED HQ, LQ_EDIT, and LQ_CLOSE. 
#WE WANT TO DESIGN A MODEL AND SEE IF IT CATEGORIZES OUR QUESTIONS INTO THE SAME CATEGORIES AS THEIR
#ACTUAL CATEGORIES
# Importing libraries
# Data Manipulation/ Handling
import pandas as pd, numpy as np
import pickle

# Visualization
import seaborn as sb, matplotlib.pyplot as plt

# NLP
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

stop_words = set(stopwords.words('english'))

# Importing dataset


reviews_df = pd.read_pickle('reviews_segment.pkl')[['review_id', 'review_text']]


# Output the DataFrame first 5 columns
print(reviews_df.head())
#print(len(df))
#print((sorted(set(df))))
print(reviews_df.review_text)
print(reviews_df.review_id[0])


# Cleaning up symbols & HTML tags
symbols = re.compile(pattern = '[/<>(){}\[\]\|@,;]')
tags = ['href', 'http', 'https', 'www']

def id_clean(s: str) -> str:
    s = s.replace("'","")
    return s

def text_clean(s: str) -> str:
    s = symbols.sub(' ', s) #s = input string, this line replaces all symbols in input string w/ ' '
    for i in tags:
        s = s.replace(i, ' ') #replace any tags in input string w/ ' '

    return ' '.join(word for word in simple_preprocess(s) if not word in stop_words) 
    #return string that contains all filtered words separated by spaces

reviews_df.iloc[:, 0] = reviews_df.iloc[:, 0].apply(id_clean) #apply processing to each row of dataset
reviews_df.iloc[:, 1] = reviews_df.iloc[:, 1].apply(text_clean) #apply processing to each row of dataset
#ds.iloc[:, 0] = ds.iloc[:, 0].apply(text_clean) #use equal sign to apply changes to original dataframe (.apply creates new dataframe)
#0 REFERS TO COLUMN CONTAINING TEXT BODY
#don't use .values because pandas is smart and will use .apply correctly

# Output the DataFrame first 5 columns
print(reviews_df.head())
#print(len(df))
#print((sorted(set(df))))
print(reviews_df.review_text)
print(reviews_df.review_id[0])

# Train & Test subsets
X_train = reviews_df.iloc[:, 1].values #assign datasets to vars
X_test= reviews_df.iloc[:, 1].values
#reshape the question categories in column 1 into a 2d array with a single column...
#-1 tells numpy to calculate number of rows needed, 1 specifies the nummber of columns we want
#need to use .values here because we are using a numpy method on a pandas object

from collections import Counter
word_counts = Counter(word for review in X_train for word in review.split())
top_words = word_counts.most_common(30000)  # Select top xk words
coverage = sum(count for word, count in top_words) / sum(word_counts.values())
print(top_words[0])
print(top_words[1])
print(f"Coverage: {coverage:.2%}") #30000 is fine length for vocab list

print(sum(word_counts.values()) / 210761) #avg review length is 66.... Will use 100 for seq length for now

word_index = dict()
for word, index in top_words:
    word_index[word] = index
print("Here is the word, one...", word_index['one'])

# Tokenization with Keras CREATE TRAINING DOCUMENTS

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, GlobalAveragePooling1D

# Setting some parameters
vocab_size = 30000 #we are saying there are 2000 unique words
glove_dim = 300
sequence_length = 300 #may be vector dimensions?

tk = Tokenizer(num_words = vocab_size, oov_token='<UNK>') #create tokenizer object, this statement specifies there should be 30000 tokens
                                       #These 30000 tokens will be the 30000 most common words in our corpus
tk.fit_on_texts(X_train) #build indexed vocabulary from words, this method actually tells the tokenizer to...
                         #break each word into a token, map each token to an integer index, keep the first 2000 indexes
                         #indexes rank words with 1 being most common
X_train = tk.texts_to_sequences(X_train) 
X_test = tk.texts_to_sequences(X_test) 
#uses index we just built to replace tokens with integers
#operating on question text

# Padding all questions with zeros


X_train_seq = pad_sequences(X_train, maxlen = sequence_length, padding = 'post') #sets all number sequences to be same length
X_test_seq = pad_sequences(X_test, maxlen = sequence_length, padding = 'post') #sets all number sequences to be same length
#padding the sequences in this way creates better model performance apparently
#We are finished processing our training data


#Import our glove embedding
#https://nlp.stanford.edu/projects/glove/

path = 'C:\\Users\\Leeha\\Program Files\\Review-Sentiment-Search\\embeddings\\glove.840B.300d.txt'
#path to the embedding
embeddings = dict()
faults = dict()
#map our vocabulary to the vectors contained in glove file

with open(path, 'r', encoding = 'utf-8') as f:
    for line in f:
        try:
        #print(line) this prints way more than 50 integers
        #each line is a word followed by 50 integers denoting its vector
        # values = line.strip().split(r"\s+")  # Remove trailing whitespace before splitting, split on spaces and tabs
        # vector = [float(v) for v in values[1:]]  # Convert each element to float
        # embeddings[values[0]] = np.array(vector, 'float32')
        #embeddings[values[0]] = np.array(values[1:], 'float32') 
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype = 'float32')
            embeddings[word] = vector
        except:
            values = line.split()
            faults[values[0]] = values[1:]




embedding_matrix = np.zeros((vocab_size, glove_dim))

#np.zeros generates an array of 0s

word_index = tk.word_index
for word, index in word_index.items():
    if index < vocab_size:
            # If the embedding for the given word exists, retrieve it and map it to the word.
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector 

#Train embedding layer
# Training the Embedding Layer & the Neural Network


model = Sequential()
#we create a sequential model, sequential = linear and simple. Will go from processing step A to processing step B etc.
model.add(Embedding(input_dim = vocab_size, output_dim = glove_dim, input_length = sequence_length, weights = [embedding_matrix], trainable = False))
#add an embedding layer to our model, which will create numeric representations of our data, making it so 
#it can be processed by our language model.
#xxinput dimensionxx is the number of rows in our embedding matrix. 1 row for each unique word in our vocab
#xxoutput dimensionxx is the number of dimensions that the output vector for each word will have
#xxinput lengthxx is the length of input sequence that the model will expect
#Due to these input, the output shape for one of our samples in (vocab_size, 128). 
#So vocab_size = 30000 so 30000 timesteps each represented by a 128D vector
#Our model has a total of 30000x128 parameters

model.add(GlobalAveragePooling1D())
#We have a vector with 30000 x 128 dimensions. We need to flatten that out  as our dense layer will expect
#A 1D vector

#model.add(Dense(units = 128, activation = None))


model.compile(optimizer = 'adam', loss = 'mse') # , metrics = ['accuracy'])
#We are configuring our model here
#It adapts the learning rate for each parameter
#metric refers to how we will measure model performance. In this case we use accuracy...
#Correct classifications / (correct classifications + incorrect classifications)

# model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
#Here we add the glove embedding layer to our model
#trainable = false means weights won't be updated while training model

model.summary()

history = model.fit(X_train_seq, X_train_seq, epochs = 1, batch_size = 512, verbose = 1)
#we are not classifying data, we are simply mapping it into semantic space, so...
#we use x_train_seq as input and output

#Save model once done training
dataset_embeddings = model.predict(X_train_seq)
np.save("dataset_embeddings.npy", dataset_embeddings)
np.save('embedding_matrix.npy', embedding_matrix)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tk, f)
#Map models to 2D space and see if we have clustering (capturing similarity between words)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate embeddings
review_embeddings = model.predict(X_train_seq[:1000])  # Use a subset for efficiency

# Reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(review_embeddings)

# Plot
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
plt.title("t-SNE Visualization of Embedding Space")
plt.show()
# #We just get a big cloud... Looks like there may be some clustering

# Example reviews
example_1 = ["the audio quality is great"]
example_2 = ["i could not figure out the settings"]
example_3 = ["the sound was very clear"]

# Tokenize and embed
example_1_seq = tk.texts_to_sequences(example_1)
example_1_seq = pad_sequences(example_1_seq, maxlen=sequence_length, padding='post')
example_2_seq = tk.texts_to_sequences(example_2)
example_2_seq = pad_sequences(example_2_seq, maxlen=sequence_length, padding='post')
example_3_seq = tk.texts_to_sequences(example_3)
example_3_seq = pad_sequences(example_3_seq, maxlen = sequence_length, padding = 'post')

embedding_1 = model.predict(example_1_seq)
embedding_2 = model.predict(example_2_seq)
embedding_3 = model.predict(example_3_seq)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embedding_1, embedding_2)
similarity2 = cosine_similarity(embedding_1, embedding_3)
print("Cosine similarity between examples 1 and 2:", similarity[0][0])
print("Cosine similarity between examples 1 and 3:", similarity2[0][0])