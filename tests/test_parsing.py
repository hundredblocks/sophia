from bs4 import BeautifulSoup

from storage.csv_storage import store_review_list, get_review_list


def test_parsing():
    with open("tests/fixtures/yelp_page.html") as f:
        doc = f.read()
    soup = BeautifulSoup(doc, 'html.parser')
    soup.find_all()
    reviews = []
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

        review_val_dom = review_dom.find_next("div", {"class": "review-content"})
        review_text = review_val_dom.find_next("p")
        review_dic["review_text"] = review_text.text

        review_score = review_val_dom.find_next("div", {"class": "i-stars"})
        title = review_score.attrs.get("title", "")
        stars = title.split(" ")[0]
        review_dic["review_score"] = stars
        reviews.append(review_dic)

    # print(reviews)
    return reviews


def test_storage():
    reviews = test_parsing()
    url = "https://www.yelp.com/biz/philz-coffee-san-mateo"
    store_review_list(reviews, url)
    print(get_review_list(url))

if __name__=="__main__":
    # test_parsing()
    test_storage()