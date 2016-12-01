import os
import pandas as pd

from review.review import Review

csv_path = "csv/"


def store_if_not_exist(review_list, review_url):
    filename = review_url.split("/")[4]
    file_path = csv_path + filename + '.csv'
    if not os.path.isfile(file_path):
        store_review_list(review_list, review_url)
    return file_path


# Always replaces
def store_review_list(review_list, review_url):
    filename = review_url.split("/")[4]
    file_path = csv_path + filename + '.csv'
    review_dic_list = [rev.as_dict() for rev in review_list]
    df = pd.DataFrame.from_dict(review_dic_list)
    df.to_csv(file_path, index=False)
    try:
        os.remove(file_path)
    except OSError:
        pass
    df.to_csv(file_path)


def get_review_list(review_url):
    filename = review_url.split("/")[4]
    file_path = csv_path + filename + '.csv'
    df = pd.DataFrame.from_csv(file_path, index_col=0)
    csv_dict = df.to_dict("records")
    review_list = [Review(rating=rev_dic['rating'], date_published=rev_dic['date_published'], description=rev_dic['description']) for rev_dic in csv_dict]
    return review_list

