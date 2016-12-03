import os
import config.config as c
from flask import Flask

import config.config as c

from parsing.parsing import get_reviews_from_url
from routes.base import base_routes

app = Flask(__name__)
app.register_blueprint(base_routes)


if __name__ == "__main__":
    c.load()
    app.run(host=os.getenv('IP', c.config['web']['ip']),port=c.config['web']['port'])
