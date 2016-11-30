import requests
import json
import math
from review.review import Review
from functools import reduce
from bs4 import BeautifulSoup
from multiprocessing import Pool
from storage.csv_storage import store_review_list, get_review_list

# TODO iterate over pages
def get_reviews_from_url(url):
    info = _get_review_info(url)
    pages = [_get_review_url_for_page(url, page) for page in range(1, _get_number_of_pages(info.get('count', 0)))]

    with Pool(20) as p:
        all_reviews_info = p.map(_get_review_info, pages)
        all_reviews_info.append(info)
        reviews = reduce(lambda acc, i: i.get('reviews', []) + acc, all_reviews_info, [])

    store_review_list(reviews, url)
    return reviews

# Private functions


def _get_number_of_pages(review_count):
    return math.ceil(review_count / 20)


def _get_review_url_for_page(url, page_number):
    return url + "?start=" + str(page_number*20)


def _get_review_info(page_url):
    doc = requests.get(page_url).text
    soup = BeautifulSoup(doc, 'html.parser')
    j = soup.findAll("script", {"type": "application/ld+json"})[0].string.strip()
    info = json.loads(j)
    reviews = [Review(yelp=r) for r in info['review']]
    return {'count': info['aggregateRating']['reviewCount'], 'rating': info['aggregateRating']['ratingValue'], 'reviews': reviews}
