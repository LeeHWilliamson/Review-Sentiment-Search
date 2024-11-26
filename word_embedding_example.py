#EXAMPLE FROM https://towardsdatascience.com/a-guide-to-word-embeddings-8a23817ab60f
#GOAL IS TO CATEGORIZE QUESTIONS INTO 3 DIFFERENT CATEGORIES BASED ON USEFULNESS
#OUR DATA CONSISTS OF QUESTION TEXT AND CATEGORY ASSIGNMENT FOR EACH QUESTION
#I.E. EACH QUESTION HAS BEEN LABELED HQ, LQ_EDIT, and LQ_CLOSE. 
#WE WANT TO DESIGN A MODEL AND SEE IF IT CATEGORIZES OUR QUESTIONS INTO THE SAME CATEGORIES AS THEIR
#ACTUAL CATEGORIES
# Importing libraries
# Data Manipulation/ Handling
import pandas as pd, numpy as np

# Visualization
import seaborn as sb, matplotlib.pyplot as plt

# NLP
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

stop_words = set(stopwords.words('english'))

# Importing dataset
dataset = pd.read_csv('train.csv')[['Body', 'Y']].rename(columns = {'Body': 'question', 'Y': 'category'})

ds = pd.read_csv('valid.csv')[['Body', 'Y']].rename(columns = {'Body': 'question', 'Y': 'category'})



# Cleaning up symbols & HTML tags
symbols = re.compile(pattern = '[/<>(){}\[\]\|@,;]')
tags = ['href', 'http', 'https', 'www']

def text_clean(s: str) -> str:
    s = symbols.sub(' ', s) #s = input string, this line replaces all symbols in input string w/ ' '
    for i in tags:
        s = s.replace(i, ' ') #replace any tags in input string w/ ' '

    return ' '.join(word for word in simple_preprocess(s) if not word in stop_words) 
    #return string that contains all filtered words separated by spaces

dataset.iloc[:, 0] = dataset.iloc[:, 0].apply(text_clean) #apply processing to each row of dataset
ds.iloc[:, 0] = ds.iloc[:, 0].apply(text_clean) #use equal sign to apply changes to original dataframe (.apply creates new dataframe)
#0 REFERS TO COLUMN CONTAINING TEXT BODY
#don't use .values because pandas is smart and will use .apply correctly


# Train & Test subsets
X_train, y_train = dataset.iloc[:, 0].values, dataset.iloc[:, 1].values.reshape(-1, 1) #assign datasets to vars
X_test, y_test = ds.iloc[:, 0].values, ds.iloc[:, 1].values.reshape(-1, 1) 
#reshape the question categories in column 1 into a 2d array with a single column...
#-1 tells numpy to calculate number of rows needed, 1 specifies the nummber of columns we want
#need to use .values here because we are using a numpy method on a pandas object

# OHE the Categorical column
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers = [('one_hot_encoder', ohe(categories = 'auto'), [0])], 
                       remainder = 'passthrough') 
#The above is creating a columntransformer that will operate on the 0th column and passthrough any other columns
#one-hot-encoder means we will create a binary feature 0-1 for each category, 0 means absence, 1 means presence
#'auto' is just saying that the transformer will automatically decide how many categories there should be

y_train = ct.fit_transform(y_train) #apply transformer to training data
y_test = ct.transform(y_test) #apply transformer to testing data

# Setting some parameters
vocab_size = 2000 #we are saying there are 2000 unique words
sequence_length = 100 #may be vector dimensions?


# Tokenization with Keras CREATE TRAINING DOCUMENTS

from tensorflow.keras.preprocessing.text import Tokenizer

tk = Tokenizer(num_words = vocab_size) #create tokenizer object, this statement specifies there should be 2000 tokens
                                       #These 2000 tokens will be the 2000 most common words in our corpus
tk.fit_on_texts(X_train) #build indexed vocabulary from words, this method actually tells the tokenizer to...
                         #break each word into a token, map each token to an integer index, keep the first 2000 indexes
                         #indexes rank words with 1 being most common
X_train = tk.texts_to_sequences(X_train) 
X_test = tk.texts_to_sequences(X_test) 
#uses index we just built to replace tokens with integers
#operating on question text

# Padding all questions with zeros
from keras.preprocessing.sequence import pad_sequences

X_train_seq = pad_sequences(X_train, maxlen = sequence_length, padding = 'post') #sets all number sequences to be same length
X_test_seq = pad_sequences(X_test, maxlen = sequence_length, padding = 'post') #sets all number sequences to be same length
#padding the sequences in this way creates better model performance apparently
#We are finished processing our training data

#Train embedding layer
# Training the Embedding Layer & the Neural Network
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten

model = Sequential()
#we create a sequential model, sequential = linear and simple. Will go from processing step A to processing step B etc.
model.add(Embedding(input_dim = vocab_size, output_dim = 5, input_length = sequence_length))
#add an embedding layer to our model, which will create numeric representations of our data, making it so 
#it can be processed by our language model.
#xxinput dimensionxx is the number of rows in our embedding matrix. 1 row for each unique word in our vocab
#xxoutput dimensionxx is the number of dimensions that the output vector for each word will have
#xxinput lengthxx is the length of input sequence that the model will expect
#Due to these input, the output shape for one of our samples in (vocab_size, 5). 
#So vocab_size = 2000 so 2000 timesteps each represented by a 5D vector
#Our model has a total of 10000 parameters

model.add(Flatten())
#We have a vector with 2000 x 5 dimensions. We need to flatten that out  as our dense layer will expect
#A 1D vector

model.add(Dense(units = 3, activation = 'softmax'))
#takes our input from before and generate a probability distribution over 3 classes
#Each input will be classified based on this probability.

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])
#We are configuring our model here
#categorical_crossentropy is a loss function that will measure the discrepency between the predicted
#probability distribution and the true distribution of the target class (how many questions are actually in each category)
#rmsprop specifies our method for updating model weight during training. Rootmeansquare propagation
#It adapts the learning rate for each parameter
#metric refers to how we will measure model performance. In this case we use accuracy...
#Correct classifications / (correct classifications + incorrect classifications)

model.summary()

history = model.fit(X_train_seq, y_train, epochs = 20, batch_size = 512, verbose = 1)

# Save model once done training
#model.save("model.h5")

#Visuzalize results

# Evaluating model performance on test set
loss, accuracy = model.evaluate(X_test_seq, y_test, verbose = 1)
print("\nAccuracy: {}\nLoss: {}".format(accuracy, loss))

# Plotting Accuracy & Loss over epochs
sb.set_style('darkgrid')

# 1) Accuracy
plt.plot(history.history['accuracy'], label = 'training', color = '#003399')
plt.legend(shadow = True, loc = 'lower right')
plt.title('Accuracy Plot over Epochs')
plt.show()

# 2) Loss
plt.plot(history.history['loss'], label = 'training loss', color = '#FF0033')
plt.legend(shadow = True, loc = 'upper right')
plt.title('Loss Plot over Epochs')
plt.show()