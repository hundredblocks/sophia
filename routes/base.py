from flask import Blueprint, render_template, request
from summary.summary import Summary


base_routes = Blueprint('requester', __name__ )


@base_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@base_routes.route('/summary', methods=['POST'])
def display_results():
    url = request.form['reviewUrl']
    summary = Summary(url)
    return render_template('summary.html',
                           rating=summary.rating(),
                           total=summary.review_count(),
                           negative_words=summary.negative_words(),
                           positive_words=summary.positive_words(),
                           summary=summary.text())
