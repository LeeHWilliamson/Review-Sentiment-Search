import pandas as pd
import pickle
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from Parse_Utils import IndexedText
from collections import defaultdict
import numpy as np
import sys

""" pancakes """

# Read the DataFrame from the pickle file
df = pd.read_pickle("reviews_segment.pkl")

# Output the DataFrame first 5 columns
print(df.head())

print(df.review_text)
print(df.review_id[0])

review_id_dict = {} #dictionary that will contain integer : review_id
for review_number, id in enumerate(df['review_id']): #iterate through every review. assign an integer key

    review_id_dict[review_number] = id.replace("'", "")
    #NOTE CAN SEARCH FOR VALUE BY USING SET AS SEARCH TERM


count = 0 #make sure dictionary looks like you'd expect
for key, value in review_id_dict.items():
    if count >= 5:
        break
    print(key, value)
    count += 1


postings_list = defaultdict(set)
porter = nltk.PorterStemmer()
stopWords = nltk.corpus.stopwords.words('english')

for review_number, text in enumerate(df['review_text']): #iterate through every review. assign an integer key

    punc = '''!()[]{};:"'-\,<>./?@#$%^&*_~''' #specify punctuation to remove
    for characters in text:
        if characters in punc:
            text = text.replace(characters, "")
    text = text.lower()
    tokenized_text = word_tokenize(text) #use NLTK to break strings into lists of tokens
    filtered_tokens = [word for word in tokenized_text if word not in stopWords] #remove tokens that are stopwords
    stemmed_tokens = [porter.stem(t) for t in filtered_tokens] #use porter stemming method
    for word in stemmed_tokens:
        postings_list[word].add(review_number)
    print(review_number, " complete")
    #NOTE CAN SEARCH FOR VALUE BY USING SET AS SEARCH TERM
for key in postings_list:
    postings_list[key] = {np.int32(idx) for idx in postings_list[key]}


with open("review_id_indexes.pkl", "wb") as f:
    pickle.dump(review_id_dict, f)
with open("postings_list.pkl", "wb") as f:
    pickle.dump(postings_list, f)

# review_id_indexes_df = pd.DataFrame.from_dict(review_id_dict, orient='index', columns=['review_id'])
# postings_list_df = pd.DataFrame.from_dict(postings_list, orient='index', columns=['review_indexes_list'])
# review_id_indexes_df.to_pickle('review_id_indexes_df.pkl')
# postings_list_df.to_pickle('postings_list_df.pkl')