import datetime
import logging
import numpy as np

import collections
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument
from matplotlib.pyplot import figure, subplot, scatter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

import gensim
from gensim import corpora, models
import multiprocessing
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import Doc2Vec, tfidfmodel
from stop_words import get_stop_words
from parsing.parsing import _get_review_info_amazon
from parsing.parsing import get_reviews_from_url

from random import sample, shuffle, random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1

def test_parsing():
    url = "https://www.amazon.com/WolVol-Walking-Triceratops-Dinosaur-Movement/product-reviews/B00M4Q2AEW"
    reviews = get_reviews_from_url(url)
    print(len(reviews))


def test_amazon_parsing():
    url = "https://www.amazon.com/WolVol-Walking-Triceratops-Dinosaur-Movement/product-reviews/B00M4Q2AEW"
    reviews = _get_review_info_amazon(url)
    print(reviews)


def test_word2vec_onlywords(reviews, model=None):
    reviews_text = [r.review['description'] for r in reviews]
    rev_nested = [nltk.tokenize.sent_tokenize(review_text) for review_text in reviews_text]
    lines_list = [line for rev_lines in rev_nested for line in rev_lines]
    print(len(reviews_text), len(rev_nested), len(lines_list))
    alldocs = []
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    words = []
    # words_dic = {}
    for i, l in enumerate(lines_list):
        subl = tokenizer.tokenize(l)
        stopped_tokens = [i for i in subl if i not in en_stop]
        words.extend(stopped_tokens)
        # for a in stopped_tokens:
        # words_dic[a]=1
        alldocs.append(stopped_tokens)
    if model is None:
        model = gensim.models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin',
                                                            binary=True)

    logging.info("loaded")
    word_vectors = []
    words = []
    for j, sent in enumerate(alldocs):
        for word in sent:
            if word in model.vocab:
                word_vectors.append(model[word])
                words.append(word)

    X_tsne = TSNE().fit_transform(word_vectors)
    logging.info("TSNE done")

    n_c = 10
    km = KMeans(n_clusters=n_c, init='k-means++', max_iter=100)
    km.fit(word_vectors)
    figure(figsize=(10, 5))
    scatter(X_tsne[:, 0], X_tsne[:, 1], c=km.labels_, alpha=0.5)
    logging.info("Clustered")
    word_counter = collections.Counter()
    for cluster in range(n_c):
        member_idexes = [memb[0] for memb in enumerate(km.labels_) if memb[1] == cluster]
        size_perc = 100 * len(member_idexes) / len(km.labels_)
        print("Cluster %s: %s percent" % (cluster, size_perc))
        # chosen_examples = sample(member_idexes, min(10, len(member_idexes)))
        chosen_words = [words[i] for i in member_idexes]
        cluster_counter = collections.Counter()
        cluster_counter.update(chosen_words)
        word_counter.update(chosen_words)
        print(cluster_counter.most_common(20))
    print(word_counter.most_common(20))


def test_word2vec(reviews, model=None):
    reviews_text = [r.review['description'] for r in reviews]
    # reviews_text = [[r.review['description'], r.review["rating"]] for r in reviews]
    rev_nested = [nltk.tokenize.sent_tokenize(review_text) for review_text in reviews_text]
    lines_list = [line for rev_lines in rev_nested for line in rev_lines]
    # TODO faster
    line_to_rev = [[line, i] for i, rev_lines in enumerate(rev_nested) for line in rev_lines]
    total_reviews = len(reviews_text)
    print(len(reviews_text), len(rev_nested), len(lines_list))
    alldocs = []
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    words = []
    for i, l in enumerate(lines_list):
        subl = tokenizer.tokenize(l)
        stopped_tokens = [i.lower() for i in subl if i not in en_stop]
        words.extend(stopped_tokens)
        alldocs.append(stopped_tokens)
    logging.info("sentences constructed")

    dictionary = corpora.Dictionary(alldocs)
    corpus = [dictionary.doc2bow(text) for text in alldocs]
    tfidf = tfidfmodel.TfidfModel(corpus)
    tfidf_scores = {dictionary.get(id): value for doc in tfidf[corpus] for id, value in doc}
    logging.info("TFIDF computed")

    sid = nltk.sentiment.SentimentIntensityAnalyzer()
    line_scores = []
    for rev_text in reviews_text:
        er = nltk.tokenize.sent_tokenize(rev_text)
        for line in er:
            line_scores.append(sid.polarity_scores(line)["compound"])
    logging.info("sentiments analyzed")

    if model is None:
        model = gensim.models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        logging.info("loaded")

    sent_vectors = []
    map_vec_to_str = {}
    i = 0
    for j, sent in enumerate(alldocs):
        total_sent_vector = sum([model[word]*tfidf_scores[word] for word in sent if word in model.vocab])
        if not isinstance(total_sent_vector, int):
            # total_weights = sum([tfidf_scores[word] for word in sent if word in model.vocab])
            map_vec_to_str[i] = j
            total_weights = len([word for word in sent if word in model.vocab])
            sent_vector = total_sent_vector/total_weights
            # TODO remove .1?
            # sent_vector_with_pol = np.append(sent_vector, .1*line_scores[map_vec_to_str[i]])
            sent_vector_with_pol = np.append(sent_vector, line_scores[j])
            rev_index = line_to_rev[j][1]
            sent_vector_with_pol_and_rev = np.append(sent_vector_with_pol, reviews[rev_index].review["rating"])
            sent_vectors.append(sent_vector_with_pol_and_rev)
            # sent_vectors.append(sent_vector_with_pol)
            i += 1

    logging.info("vectors retrieved, from %s to %s" % (len(alldocs), len(sent_vectors)))

    X_tsne = TSNE().fit_transform(sent_vectors)
    logging.info("TSNE done")

    n_c = 20
    # number_clusters = [10,15,20,50]
    results_dict = {}
    # for n_c in number_clusters:
    print("NUMBER CLUSTERS", n_c)
    km = KMeans(n_clusters=n_c, init='k-means++', max_iter=100)
    km.fit(sent_vectors)
    figure(figsize=(10, 5))
    scatter(X_tsne[:, 0], X_tsne[:, 1], c=km.labels_, alpha=0.5)
    logging.info("Clustered")
    word_counter = collections.Counter()
    cluster_config_arr = []
    for cluster in range(n_c):
        member_indexes = [memb[0] for memb in enumerate(km.labels_) if memb[1] == cluster]
        size_num =  len(member_indexes)
        size_perc = 100 * len(member_indexes) / len(km.labels_)
        cluster_counter = collections.Counter()
        translated_indexes = [map_vec_to_str[a] for a in member_indexes]

        review_indexes = [line_to_rev[a][1] for a in translated_indexes]
        num_reviews = len(set(review_indexes))
        perc_reviews = 100 * len(set(review_indexes)) / total_reviews

        sentences_indexed = [[map_vec_to_str[tr_i], sent_vectors[tr_i]] for tr_i in member_indexes]
        closest_idx = closest_to_barycenter(sentences_indexed, n=5)
        # median_positivity = np.median([a[1][-2] for a in sentences_indexed])
        avg_positivity = np.average([a[1][-2] for a in sentences_indexed])
        for ind in translated_indexes:
            cluster_counter.update([w for w in alldocs[ind] if w not in en_stop and len(w) > 1])
            word_counter.update([w for w in alldocs[ind] if w not in en_stop and len(w) > 1])

        cluster_res = {
            "CLOSEST": [lines_list[c] for c in closest_idx] if len(closest_idx)>0 else 0,
            "MOST COMMON": cluster_counter.most_common(10),
            # "MOST COMMON_MOD": most_common_uncommon(cluster_counter, tfidf_scores, n=20),
            # "NUM": cluster,
            "cluster_counter": cluster_counter,
            "PERC_SENT": size_perc,
            "NUM_SENT": size_num,
            "PERC_REV": perc_reviews,
            "NUM_REV": num_reviews,
            # "POS_MED": median_positivity,
            "POS_AVG": avg_positivity
        }
        cluster_config_arr.append(cluster_res)
    sorted_result = sorted(cluster_config_arr, key=lambda x: x["PERC_SENT"], reverse=True)
    groom_counters(sorted_result)
    print("RESULTS")
    print(sorted_result)


def groom_counters(cluster_arr):
    glob_count = collections.Counter()
    for c in cluster_arr:
        count = c["cluster_counter"]
        glob_count.update(count)
    avg_count = collections.Counter()
    per_sent_freq = {}
    total_sentences = sum([a["NUM_SENT"] for a in cluster_arr])
    n_c = len(cluster_arr)
    for key, val in glob_count.items():
        avg_count[key] = int(val/n_c)
        per_sent_freq[key] = float(val/total_sentences)

    for i, clus in enumerate(cluster_arr):
        count = clus["cluster_counter"]
        # count.subtract(avg_count)

        scaled_counts = collections.Counter()
        num_sentences = clus["NUM_SENT"]
        for key, val in per_sent_freq.items():
            scaled_counts[key] = int(val*num_sentences)
        count.subtract(scaled_counts)

        clus["GROOMED"] = [val[0] for val in count.most_common(10) if val[1] != 0]

    for cl in cluster_arr:
        del cl["cluster_counter"]


def most_common_uncommon(count: collections.Counter, tf_idf_model, n=20):
    srted = sorted([[a, count[a]*tf_idf_model[a]] for a in count.keys()], key=lambda x: x[1], reverse=True)
    return srted[0:min(len(count.keys()), n)]


def closest_to_barycenter(sentences, n=5):
    if len(sentences) == 0:
        return
    if len(sentences) == 1:
        return [sentences[0][0]]

    bary = sum([s[1] for s in sentences])/len(sentences)

    def dist(veca, vecb):
        return (sum([(veca[i]-vecb[i])**2 for i in range(len(veca))]))**.5
    mini = [300000000]*min(n,len(sentences))
    idxs = [0]*min(n,len(sentences))
    for i, sent in enumerate(sentences):
        distance = dist(sent[1], bary)
        if distance < max(mini):
            a = mini.index(max(mini))
            mini[a] = distance
            idxs[a] = sent[0]
    sorted_idxs = sorted([[a, b] for a, b in zip(idxs, mini)], key= lambda x: x[1])
    return [a[0] for a in sorted_idxs]


def test_doc2vec(reviews):
    reviews_text = [r.review['description'] for r in reviews]
    rev_nested = [nltk.tokenize.sent_tokenize(review_text) for review_text in reviews_text]
    lines_list = [line for rev_lines in rev_nested for line in rev_lines]
    print(len(reviews_text), len(rev_nested), len(lines_list))
    alldocs = []
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    en_stop = get_stop_words('en')
    p_stemmer = nltk.PorterStemmer()
    for i, l in enumerate(lines_list):
        subl = tokenizer.tokenize(l)
        stopped_tokens = [i for i in subl if not i in en_stop]
        texts = [p_stemmer.stem(i) for i in stopped_tokens]
        alldocs.append(TaggedDocument(words=texts, tags=[i]))

    # doc_list = alldocs[:]
    # simple_model = Doc2Vec(dm=1, dm_concat=1, size=300, window=5, negative=5, hs=0, min_count=2, workers=cores)
    # simple_model.build_vocab(doc_list)
    # alpha, min_alpha, passes = (0.025, 0.001, 30)
    # alpha_delta = (alpha - min_alpha) / passes
    # for epoch in range(passes):
    #     shuffle(doc_list)
    #     simple_model.alpha, simple_model.min_alpha = alpha, alpha
    #     simple_model.train(doc_list)
    #
    #     print('completed pass %i at alpha %f' % (epoch + 1, alpha))
    #     alpha -= alpha_delta
    # simple_model.save('test')

    simple_model = Doc2Vec.load('test')
    vecs = [simple_model.docvecs[i] for i in range(len(simple_model.docvecs))]

    X_tsne = TSNE().fit_transform(vecs)
    logging.info("TSNE done")

    num_clusters = [10, 30]
    for n_c in num_clusters:
        km = KMeans(n_clusters=n_c, init='k-means++', max_iter=100)
        km.fit(vecs)
        logging.info("Clustered")
        figure(figsize=(10, 5))
        scatter(X_tsne[:, 0], X_tsne[:, 1], c=km.labels_, alpha=0.5)
    # plt.legend([0], [cm])
        logging.info("Plot done")

        for cluster in range(n_c):
            word_counter = collections.Counter()
            member_idexes = [memb[0] for memb in enumerate(km.labels_) if memb[1] == cluster]
            size_perc = 100 * len(member_idexes)/len(km.labels_)
            print("Cluster %s: %s percent" % (cluster, size_perc))
            chosen_examples = sample(member_idexes, min(10, len(member_idexes)))
            cluster_counter = collections.Counter()
            for i in member_idexes:
                sent = alldocs[i][0]
                cluster_counter.update(sent)
                word_counter.update(sent)
                print(lines_list[i])

                # for val in chosen_examples:
                # print(alldocs[val])
                # print(lines_list[val])
            print("WORDS")
            print(cluster_counter.most_common(20))
            print("TOTAL")
            print(word_counter.most_common(40))


def copy_temp(revs):
    reviews_text = [r.review['description'] for r in revs]

    sid = nltk.sentiment.SentimentIntensityAnalyzer()
    print("analyzing reviews: %s" % datetime.datetime.now())
    line_scores = []
    for rev_text in reviews_text:
        lines_list = nltk.tokenize.sent_tokenize(rev_text)
        for line in lines_list:
            score = [sid.polarity_scores(line)["compound"], line]
            line_scores.append(score)
    print(line_scores)
    pos = [s[1] for s in line_scores if s[0] > 0]
    neg = [s[1] for s in line_scores if s[0] < 0]
    print("positives")
    print(pos)
    print("negatives")
    print(neg)

    pos_topics = get_topics(pos)
    neg_topics = get_topics(neg)


def get_topics(rev_text):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    raws = [rev.lower()for rev in rev_text]
    tokens = [tokenizer.tokenize(rev)for rev in raws]
    en_stop = get_stop_words('en')
    tok_array = []
    p_stemmer = nltk.PorterStemmer()
    for a, tok in enumerate(tokens):
        # print(a)
        stopped_tokens = [i for i in tok if not i in en_stop]
        texts = [p_stemmer.stem(i) for i in stopped_tokens]
        # texts = stopped_tokens
        tok_array.append(texts)
    dictionary = corpora.Dictionary(tok_array)
    print("dictionnary built")
    corpus = [dictionary.doc2bow(text) for text in tok_array]
    print("corpus built")
    print(datetime.datetime.now())
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
    print(datetime.datetime.now())
    print(ldamodel.print_topics(num_topics=10, num_words=10))
    return ldamodel.print_topics(num_topics=10, num_words=10)

if __name__=="__main__":
    # url = "https://www.yelp.com/biz/rickhouse-san-francisco"
    url = "https://www.yelp.com/biz/antoines-cookie-shop-san-mateo-3"
    review_list = get_reviews_from_url(url)
    urls = ["https://www.yelp.com/biz/antoines-cookie-shop-san-mateo-3",
            "https://www.yelp.com/biz/rickhouse-san-francisco",
            "https://www.yelp.com/biz/jougert-bar-burlingame",
            "https://www.yelp.com/biz/que-seraw-seraw-burlingame-3",
            "https://www.yelp.com/biz/taste-in-mediterranean-food-burlingame-2",
            "https://www.yelp.com/biz/diablos-jj-taqueria-burlingame",
            "https://www.yelp.com/biz/rasoi-restaurant-and-lounge-burlingame",
            "https://www.yelp.com/biz/chez-maman-san-francisco-9"]
    big_list = []
    for url in urls:
        big_list.extend(get_reviews_from_url(url))
    # test_doc2vec(get_reviews_from_url(urls[5]))

    model=gensim.models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
    # for i in range(8):
    #     test_word2vec(get_reviews_from_url(urls[i]), model=model)
    test_word2vec(get_reviews_from_url(urls[-1]), model=model)
    # test_ml(big_list)
    # test_ml(get_reviews_from_url(big_list[0:3]))
    # for i in range(6):
    #     test_ml(get_reviews_from_url(urls[i]))
    # copy_temp(get_reviews_from_url(urls[0]))
    plt.show()

