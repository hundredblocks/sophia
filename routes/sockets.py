import json
from flask_socketio import SocketIO, send
import eventlet

from parsing.parsing import get_reviews_from_url
from summary.summary import Summary

socketio = SocketIO()


@socketio.on('retrieve summary')
def summary_socket(msg_json):
    url = msg_json['url']
    reviews = get_reviews_from_url(url)
    send(json.dumps({'progress': 50}))
    summary = Summary(url, reviews=reviews)
    msg = {'progress': 100, 'summary':summary.to_dict()}
    send(json.dumps(msg))
