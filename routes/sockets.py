import json
from flask_socketio import SocketIO, send

from summary.summary import Summary

socketio = SocketIO()


@socketio.on('retrieve summary')
def summary_socket(msg_json):
    url = msg_json['url']
    summary = Summary(url)
    msg = {'progress':100, 'summary':summary.to_dict()}
    send(json.dumps(msg))
