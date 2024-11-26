import pandas as pd
import pickle
import argparse
import nltk, re, pprint
from nltk import word_tokenize
porter = nltk.PorterStemmer()

#python boolean_search_help.py -a1 audio -a2 quality -o1 poor -o2 sound -m method1 

with open("review_id_indexes.pkl", "rb") as f:
    review_id_indexes = pickle.load(f) #dic of all reviews ids
with open("postings_list.pkl", "rb") as f:
    postings_list = pd.read_pickle("postings_list.pkl")
postings_list = dict(postings_list)
print("files have been loaded")

count = 0 #make sure dictionary looks like you'd expect
for key, value in review_id_indexes.items():
    if count >= 5:
        break
    print(key, value)
    count += 1

def parse_query(query):
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




def method1(aspect1, aspect2, opinion1, opinion2):
    """
    the first method will only perform the aspect1 OR aspect2 OR opinion
    """
    print("METHOD 1 ACCESSED")
    parsed_aspect1 = parse_query(aspect1) #parse query to ensure will match lookup table
    parsed_aspect2 = parse_query(aspect2)
    parsed_opinion1 = parse_query(opinion1)
    parsed_opinion2 = parse_query(opinion2)
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
    if len(opinion_1_set) > 0:
        for word, postings in postings_list.items():
            if word == parsed_opinion2:
                print("word is...", word)
                print("postings is length", len(postings))
                opinion_2_set = postings.copy()
    if len(opinion_2_set) > 0:
        opinionIntersection = opinion_1_set.intersection(opinion_2_set)
        aspect_1_intersection = aspect_1_set.intersection(opinionIntersection)
        aspect_2_intersection = aspect_2_set.intersection(opinionIntersection)
        #termsIntersection = aspect_1_intersection.intersection(aspect_2_intersection)
        aspect_1_intersection.update(aspect_2_intersection)
        results = aspect_1_intersection
        print(len(results))
        print(results)
        
            
            
        #     for posting in postings:
        #         query_return_list.append(review_id_indexes[posting])
        # if word == parsed_aspect2:
        #     for posting in postings:
        #         query_return_list.append(review_id_indexes[posting])
        # if word == parsed_opinion1:
        #     for posting in postings:
        #         query_return_list.append(review_id_indexes[posting])

    #query_return_list = list(set(query_return_list))
    # flattened_set = set() #REMOVE DUPLICATES
    # for s in query_return_list:
    #     flattened_set.update(s)
    # query_return_list = list(flattened_set)

    # query_method1_df = pd.DataFrame(query_return_list, columns = ['review_ids'])
    # with open("query_method1.pkl", "wb") as f:
    #     pickle.dump(query_method1_df, f)
    #     print(query_method1_df.head())
    count = 0 #make sure list looks like you'd expect
    for review_id in query_return_list:
        if count >= 5:
            break
        print(review_id)
        count += 1

    return query_return_list
    pass

# def method2(aspect1, aspect2, opinion):
#     """
#     the second method will only perform the aspect1 AND aspect2 AND opinion
#     """
#     # print("Please input 2 words for the aspect you wish to query and 1 word for the opinion you wish to query")
#     # query = input("Entery 3 word query: ")

#     print("METHOD 2 ACCESSED")
#     parsed_aspect1 = parse_query(aspect1) #parse query to ensure will match lookup table
#     parsed_aspect2 = parse_query(aspect2)
#     parsed_opinion = parse_query(opinion)
#     query_return_list_aspect1 = list()
#     query_return_list_aspect2 = list()
#     query_return_list_opinion = list()
#     query_return_list_final = list()

#     for word, postings in postings_list.items():
#         if word == parsed_aspect1:
#             for posting in postings:
#                 query_return_list_aspect1.append(review_id_indexes[posting])
#                 print("aspect1 stored")
#         if word == parsed_aspect2:
#             for posting in postings:
#                 query_return_list_aspect2.append(review_id_indexes[posting])
#                 print("aspect2 stored")
#         if word == parsed_opinion:
#             for posting in postings:
#                 query_return_list_opinion.append(review_id_indexes[posting])
#                 print("opinion stored")
#     print(len(query_return_list_aspect1))
#     print(len(query_return_list_aspect2))
#     print(len(query_return_list_opinion))
#     # for review_id in query_return_list_aspect1: SHADOW
#     #     if review_id in query_return_list_aspect2:
#     #         for review_id in query_return_list_aspect1:
#     #             if review_id in query_return_list_opinion:
#     #                 query_return_list_final.append(review_id)
#     #    else:
#     #        break
#     query_return_list_final = []
#     for review_id in query_return_list_aspect1:
#         if review_id in query_return_list_aspect2 and review_id in query_return_list_opinion:
#             query_return_list_final.append(review_id)
#     count = 0 #make sure lsit looks like you'd expect
#     for review_id in query_return_list_final:
#         if count >= 5:
#             break
#         print(review_id)
#         print("TRY")
#         count += 1
#     return query_return_list_final      
#     pass

# def method3():
#     """
#     the third method will only perform the aspect1 OR aspect2 AND opinion
#     """
#     print("Please input 2 words for the aspect you wish to query and 1 word for the opinion you wish to query")
#     query = input("Entery 3 word query: ")
#     pass


def main():

    parser = argparse.ArgumentParser(description="Perform the boolean search.")
    
    parser.add_argument("-a1", "--aspect1", type=str, required=True, default=None, help="First word of the aspect")
    parser.add_argument("-a2", "--aspect2", type=str, required=True, default=None, help="Second word of the aspect")
    parser.add_argument("-o1", "--opinion1", type=str, required=True, default=None, help="First word of the opinion")
    parser.add_argument("-o2", "--opinion2", type=str, required=True, default=None, help="Second word of the opinion")
    parser.add_argument("-m", "--method", type=str, required=True, default=None, help="The method of boolean operation. Methods\
                        can be method1, method2 or method3")

    # Parse the arguments
    args = parser.parse_args()

    # review_df = pd.read_pickle("reviews_segment.pkl")
    # df = pd.read_pickle("reviews_segment.pkl")
    # review_id_indexes = pd.read_pickle("review_id_indexes_df.pkl") #df of all reviews ids
    # posting_list = pd.read_pickle("posting_list_df.pkl") #df of postings for each word

    if args.method.lower() == "method1":
        result = method1(args.aspect1.lower(), args.aspect2.lower(), args.opinion1.lower(), args.opinion2.lower())
    elif args.method.lower() == "method2":
        result = method2(args.aspect1.lower(), args.aspect2.lower(), args.opinion.lower())
    elif args.method.lower() == "method3":
        result = method3(args.aspect1.lower(), args.aspect2.lower(), args.opinion.lower())
    else:
        print("\n!! The method is not supported !!\n")
        return

    revs = pd.DataFrame()
    revs["review_index"] = [r[1:-1] for r in result] #making sure, I am not having the quotes on the index
    revs.to_pickle(args.aspect1 + "_" + args.aspect2 + "_" + args.opinion1 + "_" + args.opinion2 + "_" + args.method + ".pkl")

if __name__ == "__main__":
    main()
