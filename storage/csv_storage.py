import os
import pandas as pd

from review.review import Review

csv_path = "csv/"


def store_review_list(review_list, review_url):
    filename = review_url.split("/")[-1]
    file_path = csv_path + filename + ".csv"
    l = [r.as_dict() for r in review_list]
    df = pd.DataFrame.from_dict(l)
    df.to_csv(file_path)
    try:
        os.remove(file_path)
    except OSError:
        pass
    df = pd.DataFrame.from_dict(l)
    df.to_csv(file_path)


def get_review_list(review_url):
    filename = review_url.split("/")[-1]
    file_path = csv_path + filename + ".csv"
    df = pd.DataFrame.from_csv(file_path)
    csv_dict = df.to_dict()
    review_list = [Review(rating=csv_dict['rating'][i], date_published=csv_dict['date_published'][i], description=csv_dict['description'][i]) for i in range(len(csv_dict['description']))]
    return review_list

