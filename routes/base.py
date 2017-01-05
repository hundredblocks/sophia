import gensim
from flask import Blueprint, render_template, request, jsonify
import config.config as c

from extractor.feature_extractor import get_result
from parsing.parsing import get_reviews_from_url
from summary.summary import Summary


base_routes = Blueprint('requester', __name__ )
model = gensim.models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)


@base_routes.route('/results')
def results():
    return render_template('display_res.html')


@base_routes.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           base_route=c.config['base_url'])


@base_routes.route('/summary', methods=['POST'])
def summary():
    c.load()
    return render_template('display_res.html',
                           url=request.form['reviewUrl'],
                           base_route=c.config['base_url'])


@base_routes.route('/_get_summary')
def get_sum():
    url = request.args.get('url', '', type=str)
    revs = get_reviews_from_url(url)
    res = get_result(revs, model=model)
    return jsonify(result=res)


@base_routes.route('/api/summary', methods=['POST'])
def display_results():
    url = request.form['reviewUrl']
    summary = Summary(url)
    return render_template('summary.html',
                           rating=summary.rating(),
                           total=summary.review_count(),
                           negative_words=summary.negative_words(),
                           positive_words=summary.positive_words(),
                           summary=summary.text())


