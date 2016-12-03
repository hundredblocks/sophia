import rethinkdb as r
import config.config as c
import re


class StoreDb:

    def __init__(self):
        pass

    def save(self, doc):
        if doc.get('date_created', None) is None:
            doc['date_created'] = r.now()

        r.db(c.config['db']['name'])\
            .table(self._sanitize(self.__class__.__name__))\
            .insert(doc)\
            .run(r.connect(c.config['db']['host'], c.config['db']['port']))

    def get(self, key):
        cursor = r.db(c.config['db']['name'])\
            .table(self._sanitize(self.__class__.__name__))\
            .filter(r.row['key'] == key)\
            .run(r.connect(c.config['db']['host'], c.config['db']['port']))
        doc = None
        try:
            raw_doc = cursor.next()
            doc = self._parse(raw_doc)
        except r.ReqlCursorEmpty:
            pass

        return doc

    @staticmethod
    def _parse(raw_doc):
        return raw_doc

    @staticmethod
    def _sanitize(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
