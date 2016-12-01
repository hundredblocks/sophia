class Review:

    def __init__(self, **kwargs):
        self.review = {}
        self._parse_yelp_review(kwargs.get('yelp'))

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
        self.review['date_ublished'] = yelp_json['datePublished']
        self.review['description'] = yelp_json['description']
