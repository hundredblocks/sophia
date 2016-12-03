from parsing.parsing import get_reviews_from_url
from extractor.feature_extractor import extract
from storage.rethinkdb_storage import save, get

class Summary:

    def __init__(self, key):
        summary = get(key)
        if len(summary['words']) == 0:
            reviews = get_reviews_from_url(key)
            summary = extract(reviews, 100)
            save(key, summary)

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

    def _percent(self, l):
        l['median_positivity'] = int(round(l['median_positivity'] * 100, 0))
        return l