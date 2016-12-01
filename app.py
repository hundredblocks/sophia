import os
from flask import Flask

from parsing.parsing import get_reviews_from_url
from routes.base import base_routes

app = Flask(__name__)
app.register_blueprint(base_routes)


if __name__ == "__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)))
