from flask import Blueprint, render_template, request
from parsing.parsing import get_reviews_from_url
from storage.csv_storage import store_review_list

base_routes = Blueprint('requester', __name__ )


@base_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@base_routes.route('/summary', methods=['POST'])
def display_results():
    url = request.form['reviewUrl']
    reviews = get_reviews_from_url(url)
    return render_template('summary.html', rating=4, total=len(reviews), summary="None")
