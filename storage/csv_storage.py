import os
import pandas as pd

csv_path = "csv/"


def store_review_list(review_list, review_url):
    file_path = csv_path + review_url
    df = pd.DataFrame.from_dict(review_list)
    df.to_csv(file_path)
    try:
        os.remove(file_path)
    except OSError:
        pass
    df = pd.DataFrame.from_dict(review_list)
    df.to_csv(file_path)
    return file_path


def get_review_list(review_url):
    file_path = csv_path + review_url
    df = pd.DataFrame.from_csv(file_path)
    review_list = df.to_dict()
    return review_list

