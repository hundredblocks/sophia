import pandas as pd
import collections
from nltk.sentiment import SentimentIntensityAnalyzer
from storage.csv_storage import get_review_list
# from nltk import tokenize
import nltk



def extract(url):
    rev = get_review_list(url)

    sid = nltk.sentiment.SentimentIntensityAnalyzer()
    comp_array = []
    word_dic = collections.defaultdict(list)
    for review in rev:
        rev_text = review.description()
        # print(rev_text)
        lines_list = nltk.tokenize.sent_tokenize(rev_text)
        # scores = [[sid.polarity_scores(sentence)["compound"], sentence] for sentence in lines_list]
        for line in lines_list:
            score = [sid.polarity_scores(line)["compound"], line]
            comp_array.append(score)
            words_list = nltk.tokenize.word_tokenize(line)
            for word in words_list:
                word_dic[word].append(sid.polarity_scores(line)["compound"])

    a = sorted(comp_array, key=lambda x: x[0])
    print(a)
    # print(word_dic)
    averages = []
    for key, value in word_dic.items():
        avg = float(sum(value)/len(value))
        averages.append({"word": key,
                         "positivity": avg,
                         "count": len(value)})
    b = pd.DataFrame.from_dict(averages)
    # b = sorted(averages, key=lambda x: x[1])
    b.sort(columns=["positivity", "count"], inplace=True)
    print(b)
    print(b[b["count"]>10])

if __name__ == "__main__":
    test_url = "https://www.yelp.com/biz/farina-pizza-and-cucina-italiana-san-francisco"
    extract(test_url)