import requests
import json
import math
import re
from review.review import Review
from functools import reduce
from bs4 import BeautifulSoup
from multiprocessing import Pool
from storage.csv_storage import store_review_list, get_review_list


def get_reviews_from_url(url):
    try:
        l = get_review_list(url)
        return l
    except OSError:
        reviews = []
        if 'yelp' in url:
            info = _get_review_info(url)
            pages = [_get_review_url_for_page(url, page) for page in range(1, _get_number_of_pages(info.get('count', 0), 20))]

            with Pool(20) as p:
                all_reviews_info = p.map(_get_review_info, pages)
                all_reviews_info.append(info)
                reviews = reduce(lambda acc, i: i.get('reviews', []) + acc, all_reviews_info, [])
        elif 'amazon' in url:
            info = _get_review_info_amazon(url)
            pages = [_get_review_url_for_page_amazon(url, page) for page in range(2, min(_get_number_of_pages(info.get('count', 0), 10), 4))]

            with Pool(1) as p:
                all_reviews_info = p.map(_get_review_info_amazon, pages)
                all_reviews_info.append(info)
                reviews = reduce(lambda acc, i: i.get('reviews', []) + acc, all_reviews_info, [])

        store_review_list(reviews, url)
        return reviews

# Private functions
def _get_number_of_pages(review_count, per_page):
    return math.ceil(review_count / per_page)


def _get_review_url_for_page(url, page_number):
    return url + "?start=" + str(page_number*20)


def _get_review_info(page_url):
    doc = requests.get(page_url).text
    soup = BeautifulSoup(doc, 'html.parser')
    j = soup.findAll("script", {"type": "application/ld+json"})[0].string.strip()
    info = json.loads(j)
    reviews = [Review(yelp=r) for r in info['review']]
    return {'count': info['aggregateRating']['reviewCount'], 'rating': info['aggregateRating']['ratingValue'], 'reviews': reviews}


def _get_review_url_for_page_amazon(url, page_number):
    return url + "?pageNumber=" + str(page_number)


def _get_review_info_amazon(page_url):
    doc = requests.get(page_url).text
    soup = BeautifulSoup(doc, 'html.parser')
    reviews_dom = soup.findAll('div', {'class': 'review'})

    count = re.match(r'.*\d+-\d+ of (\d+)', soup.find('div', {'id': 'cm_cr-review_list'}).find_next('span', {'class': 'a-size-base'}).string).group(1)

    reviews = []
    for review_dom in reviews_dom:
        rating_dom = review_dom.find_next('i', {'class': 'a-icon-star'})
        rating = re.match(r'.*a-star-(\d)', ' '.join(rating_dom['class'])).group(1)

        date_published = review_dom.find_next('span', 'review-date').string.replace('on ', '')

        description = review_dom.find_next('span', {'class': 'review-text'}).string

        reviews.append(Review(rating=rating, date_published=date_published, description=description))

    return {'count': int(count), 'reviews': reviews}
