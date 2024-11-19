import pandas as pd
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from Parse_Utils import IndexedText
""" pancakes """

# Read the DataFrame from the pickle file
df = pd.read_pickle("reviews_segment.pkl")

# Output the DataFrame first 5 columns
print(df.head())
#print(len(df))
#print((sorted(set(df))))
print(df.review_text)
print(df.review_id[0])
# print(df.review_title)
review_id_dict = {} #dictionary that will contain integer : review_id
for review_number, id in enumerate(df['review_id']): #iterate through every review. assign an integer key
    #review_id_dict[review_number] = set() #intialize value in key : value to an empty set
    #review_id_dict[review_number].add(id)
    #review_id_dict[review_number].add(id.replace("'", "")) #add id string to empty set, remove '' which are pkl artifacts
    review_id_dict[review_number] = id.replace("'", "")
    #NOTE CAN SEARCH FOR VALUE BY USING SET AS SEARCH TERM

# keyList = list(review_id_dict.keys())
# for key in keyList:
#     temp = review_id_dict[key]
#     print("'" in temp)
count = 0 #make sure dictionary looks like you'd expect
for key, value in review_id_dict.items():
    if count >= 5:
        break
    print(key, value)
    count += 1

review_text_dict = {}
porter = nltk.PorterStemmer()
stopWords = nltk.corpus.stopwords.words('english')
# for review_id in df['review_id']:
#     review_text_dict[review_id] = None
for review_number, text in enumerate(df['review_text']): #iterate through every review. assign an integer key
    #review_id_dict[review_number] = set() #intialize value in key : value to an empty set
    #review_id_dict[review_number].add(id)
    punc = '''!()[]{};:"'-\,<>./?@#$%^&*_~''' #specify punctuation to remove
    for characters in text:
        if characters in punc:
            text = text.replace(characters, "")
    #text = text[1:-1] NO LONGER NEED BECAUSE I AM SIMPLY REMOVING ' FROM REVIEWS AND SEARCH QUERIES
    text = text.lower()
    #text = IndexedText(porter, text)
    tokenized_text = word_tokenize(text) #use NLTK to break strings into lists of tokens
    filtered_tokens = [word for word in tokenized_text if word not in stopWords] #remove tokens that are stopwords
    stemmed_tokens = [porter.stem(t) for t in filtered_tokens] #use porter stemming method
    review_text_dict[review_number] = stemmed_tokens #.replace("'", "") #add id string to empty set, remove '' which are pkl artifacts
    #NOTE CAN SEARCH FOR VALUE BY USING SET AS SEARCH TERM
    
count = 0 #make sure dictionary looks like you'd expect
for key, value in review_text_dict.items():
    if count >= 5:
        break
    print(key, value)
    count += 1

review_word_dict = {}
word_number = 0
for key, value in review_text_dict.items():
    for token in value:
        if token not in review_word_dict:
            review_word_dict[word_number] = token
            word_number += 1

count = 0 #make sure dictionary looks like you'd expect
for key, value in review_word_dict.items():
    if count >= 100:
        break
    print(key, value)
    count += 1

posting_list = {}
# for key, value in review_text_dict.items():
#     review_key = key
#     for token in value:
#         for key, word in review_word_dict.items():
#             if token == word and token not in inverted_index:
#                 inverted_index[key] = list()
#                 inverted_index[key].append(review_id_dict[key])
#             elif token == word and token in inverted_index:
#                 inverted_index[key].append(token)
#             else: print("error: word not found in vocabulary")
count = 0
for key, vocab_value in review_word_dict.items(): #iterate through vocab list
    posting_list[key] = list() #each iteration, initialize a key for inverted_index that matches vocab key
    for index, reviews in review_text_dict.items(): #for ech word in vocab list, iterate through all items of reviews_text by their index
            for word in reviews: #iterate through each word in each review
                if word == vocab_value and len(posting_list[key]) == 0: #if we happen upon a word that matches that of our current key in vocab list, append current review index to inverted_index
                    #inverted_index[key] = list()
                    posting_list[key].append(index)
                elif word == vocab_value and index not in posting_list[key]:
                    posting_list[key].append(index)
    posting_list[key] = list(set(posting_list[key]))
    print("entry added to inverted index")

count = 0 #make sure dictionary looks like you'd expect
for key, value in posting_list.items():
    if count >= 10:
        break
    print(key, value)
    count += 1

review_id_indexes_df = pd.DataFrame.from_dict(review_id_dict)
review_vocabulary_df = pd.DataFrame.from_dict(review_word_dict)
posting_list_df = pd.DataFrame.from_dict(posting_list)
review_id_indexes_df.to_pickle('review_id_indexes_df.pkl')
review_vocabulary_df.to_pickle('review_vocabulary_df.pkl')
posting_list_df.to_pickle('posting_list_df.pkl')