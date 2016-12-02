import datetime
import numpy as np

import pandas as pd
import collections
from nltk.sentiment import SentimentIntensityAnalyzer
from storage.csv_storage import get_review_list
# from nltk import tokenize
import nltk



def extract(url, num_reviews):
    rev = get_review_list(url)
    sid = nltk.sentiment.SentimentIntensityAnalyzer()
    comp_array = []
    word_dic = collections.defaultdict(list)
    print("analyzing reviews: %s" % datetime.datetime.now())
    print(len(rev))
    num_reviews = min(num_reviews, len(rev))
    for review in rev[0:num_reviews]:
        rev_text = review.description()
        # print(rev_text)
        lines_list = nltk.tokenize.sent_tokenize(rev_text)
        # scores = [[sid.polarity_scores(sentence)["compound"], sentence] for sentence in lines_list]
        for line in lines_list:
            score = [sid.polarity_scores(line)["compound"], line]
            comp_array.append(score)
            words_list = nltk.tokenize.word_tokenize(line)
            pos = nltk.pos_tag(words_list)
            for i, word in enumerate(words_list):
                word_dic[word].append([pos[i][1], sid.polarity_scores(line)["compound"]])

    a = sorted(comp_array, key=lambda x: x[0])
    # print(a)
    # print(word_dic)
    print("looking at results: %s" % datetime.datetime.now())
    averages = []
    for key, val in word_dic.items():
        value = [v[1] for v in val]
        # print([v[0]for v in val])
        word_type = val[0][0]
        avg = float(sum(value)/len(value))
        median = np.median(value)
        averages.append({"word": key,
                         "type": word_type,
                         "avg_positivity": avg,
                         "median_positivity": median,
                         "count": len(value)})
    word_scores = pd.DataFrame.from_dict(averages)
    # b = sorted(averages, key=lambda x: x[1])
    word_scores.sort(columns=["median_positivity", "count"], inplace=True)
    # print(b[b["type"]=="NN"])
    # print(b[b["type"]=="NP"])
    # print(b[b["type"]=="JJ"])
    # print(b[b["type"]=="RB"])
    freq_threshold = num_reviews*.05
    frequent_words = word_scores[word_scores["count"] > freq_threshold]
    print(set(frequent_words.type.values))
    for val in set(frequent_words.type.values):
        print(val)
        print(frequent_words[frequent_words["type"] == val])
    # print(b[b["count"]>10])

if __name__ == "__main__":
    test_url = "https://www.yelp.com/biz/farina-pizza-and-cucina-italiana-san-francisco"
    print(datetime.datetime.now())
    extract(test_url, 100)
    print(datetime.datetime.now())