import pandas as pd
from storage.csv_storage import get_review_list


def extract():
    test_url = "https://www.yelp.com/biz/philz-coffee-san-mateo"
    rev = get_review_list(test_url)
    df = pd.DataFrame.from_dict(rev)
    print(df.describe())


if __name__ == "__main__":
    extract()