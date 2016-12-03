import config.config as c
from flask import Flask
from routes.sockets import socketio

import config.config as c

from routes.base import base_routes

app = Flask(__name__)
app.register_blueprint(base_routes)


if __name__ == "__main__":
    c.load()
    socketio.init_app(app)
    socketio.run(app, host=c.config['web']['ip'], port=c.config['web']['port'])
