from parsing.parsing import get_reviews_from_url
from extractor.feature_extractor import extract
from storage.rethinkdb_storage import StoreDb
from review.review import Review


class Summary(StoreDb):

    def __init__(self, key):
        super().__init__()

        self.key = key
        summary = self.get(key)
        if summary is None:
            reviews = get_reviews_from_url(key)
            summary = extract(reviews, 100)
            summary['key'] = key
            summary['reviews'] = [review.as_dict() for review in reviews]
            self.save(summary)

        self._summary = summary

    def words(self):
        return [self._percent(w) for w in self._summary['words'] if w['median_positivity'] != 0]

    def negative_words(self, number=3):
        words_list = [self._percent(w) for w in self._summary['words'] if w['median_positivity'] < 0]
        return words_list[:number]

    def positive_words(self, number=3):
        words_list = [self._percent(w) for w in self._summary['words'] if w['median_positivity'] > 0]
        return words_list[:number]

    def review_count(self):
        return self._summary['review_count']

    def text(self):
        return 'This place is good!'

    def rating(self):
        return 4

    def to_dict(self):
        return {
            'words': self.words(),
            'negative_words': self.negative_words(),
            'positive_words': self.positive_words(),
            'review_count': self.review_count(),
            'text': self.text(),
            'rating': self.rating()
        }

    def _percent(self, l):
        newL = dict(l)
        newL['median_positivity'] = int(round(l['median_positivity'] * 100, 0))
        return newL

    @staticmethod
    def _parse(raw_doc):
        raw_doc['reviews'] = [Review(raw_review=review) for review in raw_doc.get('reviews', [])]
        return raw_doc
