from flask import Blueprint
from parsing.parsing import get_reviews_from_url
base_routes = Blueprint('requester', __name__ )


@base_routes.route("/")
def index():
    return "This is the landing page"


@base_routes.route("/results")
def display_results():
    url = "https://www.yelp.com/biz/philz-coffee-san-mateo"
    get_reviews_from_url(url)
    return "These are the results"