import requests
from bs4 import BeautifulSoup


def get_reviews_from_url(url):
    doc = requests.get(url).text
    soup = BeautifulSoup(doc, 'html.parser')
    soup.find_all()
    reviews_dom = soup.findAll("div", {"class": "review review--with-sidebar"})
    for review_dom in reviews_dom:
        review_dic = {}
        user_name_dom = review_dom.find_next("a", {"class": "user-display-name"})
        review_dic["user_name"] = user_name_dom.string

        user_loc_dom = review_dom.find_next("a", {"class": "user-location"})
        review_dic["user_location"] = user_loc_dom.string

        user_friend_dom = review_dom.find_next("a", {"class": "friend-count"})

        review_dic["user_friends"] = user_friend_dom.find_next("b").text

        print(review_dic)


