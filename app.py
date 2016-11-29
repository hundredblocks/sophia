from flask import Flask

from parsing.parsing import get_reviews_from_url
from routes.base import base_routes

app = Flask(__name__)
app.register_blueprint(base_routes)


if __name__ == "__main__":
    app.run()
    url = "https://www.yelp.com/biz/philz-coffee-san-mateo?osq=philz+san+mateo"
    get_reviews_from_url(url)