import pandas as pd
import pickle
import pandas as pd
import pickle
import nltk, re, pprint
from nltk import word_tokenize
porter = nltk.PorterStemmer()

reviews_df = pd.read_pickle('reviews_segment.pkl')[['review_id', 'review_text']]

with open("review_id_indexes.pkl", "rb") as f:
    review_id_indexes = pickle.load(f) #dic of all reviews ids
review_ids = dict(review_id_indexes)
with open("postings_list.pkl", "rb") as f:
    postings_list = pd.read_pickle("postings_list.pkl")
postings_list = dict(postings_list)
print("files have been loaded")

count = 0 #make sure review_list looks like you'd expect
for key, value in review_ids.items():
    if count >= 5:
        break
    print(key, value)
    count += 1

count = 0 #make sure postings_list looks like you'd expect
for key, value in postings_list.items():
    if count >= 5:
        break
    print(key, len(value))
    count += 1
print(len(postings_list.items()))

def id_clean(s: str) -> str:
    s = s.replace("'","")
    return s

reviews_df.iloc[:, 0] = reviews_df.iloc[:, 0].apply(id_clean) #apply processing to each row of dataset
reviews_df.iloc[:, 1] = reviews_df.iloc[:, 1] #apply processing to each row of dataset
print("the key type is...", type(reviews_df.keys()))
print(reviews_df.iloc[0])


def parse_query(query):
    if query != None:
        punc = '''!()[]{};:"'-\,<>./?@#$%^&*_~''' #specify punctuation to remove
        print(query)
        print("QUERY IS ABOVE")
        print(type(query))
        print(type(postings_list))
        for characters in query:
            if characters in punc:
                query = query.replace(characters, "")
        query = query.lower()
        query = word_tokenize(query)
        query = [porter.stem(t) for t in query]
        print(type(query[0]))
        return(query[0])
    else:
        return query


print("METHOD 1 ACCESSED")
parsed_aspect1 = parse_query("audio") #parse query to ensure will match lookup table
parsed_aspect2 = parse_query("quality")
parsed_opinion1 = parse_query("good")
parsed_opinion2 = parse_query(None)
aspect_1_set = set()
aspect_2_set = set()
opinion_1_set = set()
opinion_2_set = set()
query_return_list = list()

for word, postings in postings_list.items():
    if word == parsed_aspect1:
        print("word is...", word)
        print("postings is length", len(postings))
        aspect_1_set = postings.copy()
if len(aspect_1_set) > 0:
    for word, postings in postings_list.items():
        if word == parsed_aspect2:
            print("word is...", word)
            print("postings is length", len(postings))
            aspect_2_set = postings.copy()
if len(aspect_2_set) > 0:
    for word, postings in postings_list.items():
        if word == parsed_opinion1:
            print("word is....", word)
            print("postings is length", len(postings))
            opinion_1_set = postings.copy()
if len(opinion_1_set) > 0 and parsed_opinion2 != None:
    for word, postings in postings_list.items():
        if word == parsed_opinion2:
            print("word is...", word)
            print("postings is length", len(postings))
            opinion_2_set = postings.copy()
if parsed_opinion2 == None:
    opinionIntersection = opinion_1_set # .intersection(opinion_2_set)
    aspect_1_intersection = aspect_1_set.intersection(opinionIntersection)
    aspect_2_intersection = aspect_2_set.intersection(opinionIntersection)
    #termsIntersection = aspect_1_intersection.intersection(aspect_2_intersection)
    aspect_1_intersection.update(aspect_2_intersection)
    results = aspect_1_intersection
    print(len(results))
    results_indexes = []
    #print(results)
    count = 0
    for item in results: #print review indexes
        if count < 5:
            print(item)
            results_indexes.append(item)
            #print(results_list[count])
            count += 1
        else:
            break
    results_reviews_ids = []
    count = 0
    for index in results_indexes:
        if count < 5:
            print(count)
            results_reviews_ids.append(review_ids[(results_indexes[count])])
            print("review_id returned is...", results_reviews_ids[count])
            count += 1
    count = 0 
    for id in results_reviews_ids:
        if count < 5:
           # print(type(reviews_df))
           # print("this id does exist, see?...", reviews_df.loc[reviews_df['review_id'] == 'RQXIKJIYCUOS3', 'review_text'].values[0])
           print(reviews_df.loc[reviews_df['review_id'] == results_reviews_ids[count], 'review_text'].values[0])
           count += 1
           # print("NEXT REVIEW")
else:
    opinionIntersection = opinion_1_set.intersection(opinion_2_set)
    aspect_1_intersection = aspect_1_set.intersection(opinionIntersection)
    aspect_2_intersection = aspect_2_set.intersection(opinionIntersection)
    #termsIntersection = aspect_1_intersection.intersection(aspect_2_intersection)
    aspect_1_intersection.update(aspect_2_intersection)
    results = aspect_1_intersection
    print(len(results))
    #print(results)