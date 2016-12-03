import rethinkdb as r

from review.review import Review


db_name = 'sophia'
table_name = 'summary'


def save(key, summary):
    doc = {
        'key': key,
        'words': summary.get('words', []),
        'review_count': summary.get('review_count', 0),
        'reviews': [review.as_dict() for review in summary.get('reviews', [])],
        'date_created': r.now()
    }
    r.db(db_name).table(table_name).insert(doc).run(r.connect('localhost', 28015))


def get(key):
    cursor = r.db(db_name).table(table_name).filter(r.row['key'] == key).run(r.connect('localhost', 28015))
    summary = {'words': [], 'count': 0}
    try:
        results = cursor.next()
        summary = results
        summary['reviews'] = [Review(raw_review=review) for review in summary.get('reviews', [])]
    except r.ReqlCursorEmpty:
        pass

    return summary
