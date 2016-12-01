from parsing.parsing import get_reviews_from_url, _get_review_info_amazon

from storage.csv_storage import store_review_list, get_review_list


def test_parsing():
    url = "https://www.amazon.com/WolVol-Walking-Triceratops-Dinosaur-Movement/product-reviews/B00M4Q2AEW"
    reviews = get_reviews_from_url(url)
    print(len(reviews))


def test_amazon_parsing():
    url = "https://www.amazon.com/WolVol-Walking-Triceratops-Dinosaur-Movement/product-reviews/B00M4Q2AEW"
    reviews = _get_review_info_amazon(url)
    print(reviews)


def test_storage():
    reviews = test_parsing()
    url = "https://www.yelp.com/biz/philz-coffee-san-mateo"
    store_review_list(reviews, url)
    print(get_review_list(url))

if __name__=="__main__":
    test_parsing()
    # test_storage()
