from parsing.parsing import get_reviews_from_url

from storage.csv_storage import store_review_list, get_review_list


def test_parsing():
    url = "https://www.yelp.com/biz/philz-coffee-san-mateo"
    reviews = get_reviews_from_url(url)
    print(reviews)


def test_storage():
    reviews = test_parsing()
    url = "https://www.yelp.com/biz/philz-coffee-san-mateo"
    store_review_list(reviews, url)
    print(get_review_list(url))

if __name__=="__main__":
    test_parsing()
    # test_storage()
