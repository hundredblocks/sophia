class Review:

    def __init__(self, **kwargs):
        self.review = {}
        yelp_json = kwargs.get('yelp')
        raw_review = kwargs.get('review')

        if yelp_json is not None:
            self._parse_yelp_review(kwargs.get('yelp'))
        elif raw_review is not None:
            self._parse_raw_review(raw_review)
        else:
            self.review['rating'] = kwargs.get('rating')
            self.review['date_published'] = kwargs.get('date_published')
            self.review['description'] = kwargs.get('description')

    def rating(self):
        return self.review.get('rating')

    def description(self):
        return self.review.get('description')

    def date_published(self):
        return self.review.get('data_published')

    def as_dict(self):
        return self.review

    # Private methods
    def _parse_yelp_review(self, yelp_json):
        self.review['rating'] = yelp_json['reviewRating']['ratingValue']
        self.review['date_published'] = yelp_json['datePublished']
        self.review['description'] = yelp_json['description']

    def _parse_raw_review(self, raw_review):
        self.review = raw_review
