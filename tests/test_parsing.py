from bs4 import BeautifulSoup


def test_parsing():
    with open("fixtures/yelp_page.html") as f:
        doc = f.read()
    soup = BeautifulSoup(doc, 'html.parser')
    soup.find_all()
    reviews_dom = soup.findAll("div", {"class": "review review--with-sidebar"})
    for review_dom in reviews_dom:
        review_dic = {}
        user_name_dom = review_dom.find_next("a", {"class": "user-display-name"})
        review_dic["user_name"] = user_name_dom.string

        user_loc_dom = review_dom.find_next("li", {"class": "user-location responsive-hidden-small"}).find_next("b")
        review_dic["user_location"] = user_loc_dom.string

        user_friend_dom = review_dom.find_next("li", {"class": "friend-count"}).find_next("b")
        review_dic["user_friends"] = user_friend_dom.text

        user_reviewcount_dom = review_dom.find_next("li", {"class": "review-count"}).find_next("b")
        review_dic["user_reviews"] = user_reviewcount_dom.text


if __name__=="__main__":
    test_parsing()