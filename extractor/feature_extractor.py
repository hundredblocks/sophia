import logging
import collections
import json
import multiprocessing
import copy

import numpy as np
from sklearn.cluster import KMeans

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import gensim
from gensim import corpora
from gensim.models import tfidfmodel
from stop_words import get_stop_words

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1


def get_result(reviews, n_c=20, model=None):
    reviews_text = [r.review['description'] for r in reviews]
    en_stop = get_stop_words('en')
    total_reviews = len(reviews_text)
    line_to_rev, lines_list, alldocs, words = prepare_review_data(reviews_text)

    tfidf_scores = get_tf_idf(alldocs)

    line_scores = get_sentiment_scores(reviews_text)

    if model is None:
        model = gensim.models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
        logging.info("loaded")

    sent_vectors, map_vec_to_str = get_sentence_vectors(alldocs, model, tfidf_scores, reviews, line_scores, line_to_rev)
    logging.info("vectors retrieved, from %s to %s" % (len(alldocs), len(sent_vectors)))

    km = get_clusters(sent_vectors, n_c=n_c)

    cluster_summaries = get_cluster_summaries(km, sent_vectors, lines_list, alldocs, map_vec_to_str, line_to_rev,
                                              total_reviews, en_stop, n_c=n_c)
    sorted_summaries = sorted(cluster_summaries, key=lambda x: x["PERC_SENT"], reverse=True)

    groom_counters(sorted_summaries)
    pick_sentences(sorted_summaries, n=5)
    clean_summaries = clean_clusters(sorted_summaries)

    json_result = json.dumps(clean_summaries)
    logging.info(json_result)
    return json_result


def get_cluster_summaries(km, sent_vectors, lines_list, alldocs, map_vec_to_str, line_to_rev, total_reviews, en_stop,
                          n_c=20):
    word_counter = collections.Counter()
    cluster_summaries = []
    for cluster in range(n_c):
        member_indexes = [memb[0] for memb in enumerate(km.labels_) if memb[1] == cluster]
        size_num = len(member_indexes)
        size_perc = 100 * len(member_indexes) / len(km.labels_)
        perc_reviews = 0
        num_reviews = 0
        avg_positivity = 0
        cluster_counter = collections.Counter()
        closest_idx = []
        translated_indexes = []
        if size_num > 0:
            translated_indexes = [map_vec_to_str[a] for a in member_indexes]

            review_indexes = [line_to_rev[a][1] for a in translated_indexes]
            num_reviews = len(set(review_indexes))
            perc_reviews = 100 * len(set(review_indexes)) / total_reviews

            sentences_indexed = [[map_vec_to_str[tr_i], sent_vectors[tr_i]] for tr_i in member_indexes]
            closest_idx = closest_to_barycenter(sentences_indexed, n=5)
            avg_positivity = np.average([a[1][-2] for a in sentences_indexed])
            for ind in translated_indexes:
                cluster_counter.update([w for w in alldocs[ind] if w not in en_stop and len(w) > 1])
                word_counter.update([w for w in alldocs[ind] if w not in en_stop and len(w) > 1])

        cluster_res = {
            "CLOSEST": [lines_list[c] for c in closest_idx] if len(closest_idx) > 0 else 0,
            "MOST COMMON": cluster_counter.most_common(10),
            "ID": cluster,
            "cluster_counter": cluster_counter,
            "sentences": [lines_list[ind] for ind in translated_indexes],
            "NUM_SENT": size_num,
            "PERC_SENT": size_perc,
            "PERC_REV": perc_reviews,
            "NUM_REV": num_reviews,
            "POS_AVG": avg_positivity
        }
        cluster_summaries.append(cluster_res)
    return cluster_summaries


def get_clusters(sent_vectors, n_c=20):
    km = KMeans(n_clusters=n_c, init='k-means++', max_iter=100)
    km.fit(sent_vectors)
    logging.info("Clustered in %s clusters" % n_c)
    return km


def get_sentence_vectors(alldocs, model, tfidf_scores, reviews, line_scores, line_to_rev):
    sent_vectors = []
    map_vec_to_str = {}
    i = 0
    for j, sent in enumerate(alldocs):
        total_sent_vector = sum([model[word] * tfidf_scores[word] for word in sent if word in model.vocab])
        if not isinstance(total_sent_vector, int):
            map_vec_to_str[i] = j
            total_weights = len([word for word in sent if word in model.vocab])
            sent_vector = total_sent_vector / total_weights
            sent_vector_with_pol = np.append(sent_vector, line_scores[j])
            rev_index = line_to_rev[j][1]
            sent_vector_with_pol_and_rev = np.append(sent_vector_with_pol, reviews[rev_index].review["rating"])
            sent_vectors.append(sent_vector_with_pol_and_rev)
            i += 1
    return sent_vectors, map_vec_to_str


def get_tf_idf(alldocs):
    dictionary = corpora.Dictionary(alldocs)
    corpus = [dictionary.doc2bow(text) for text in alldocs]
    tfidf = tfidfmodel.TfidfModel(corpus)
    tfidf_scores = {dictionary.get(id): value for doc in tfidf[corpus] for id, value in doc}
    logging.info("TFIDF computed")
    return tfidf_scores


def get_sentiment_scores(reviews_text):
    sid = nltk.sentiment.SentimentIntensityAnalyzer()
    line_scores = []
    for rev_text in reviews_text:
        er = nltk.tokenize.sent_tokenize(rev_text)
        for line in er:
            line_scores.append(sid.polarity_scores(line)["compound"])
    logging.info("sentiments analyzed")
    return line_scores


def prepare_review_data(reviews_text):
    rev_nested = [nltk.tokenize.sent_tokenize(review_text) for review_text in reviews_text]
    lines_list = [line for rev_lines in rev_nested for line in rev_lines]
    line_to_rev = [[line, i] for i, rev_lines in enumerate(rev_nested) for line in rev_lines]
    total_reviews = len(reviews_text)
    logging.info("%s reviews, %s lines" % (total_reviews, len(lines_list)))
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
    return line_to_rev, lines_list, alldocs, words


def clean_clusters(cluster_arr):
    new_clus_arr = []
    for clus in cluster_arr:
        new_clus = {
            "ID": clus["ID"],
            "NUM_SENT": clus["NUM_SENT"],
            "PERC_REV": clus["PERC_REV"],
            "PERC_SENT": clus["PERC_SENT"],
            "NUM_REV": clus["NUM_REV"],
            "POS_AVG": clus["POS_AVG"],
            "GROOMED": clus["GROOMED"],
            "GROOMED_COUNT": clus["GROOMED_COUNT"],
            "groom": clus["groom"],
            "CHOSEN": clus["CHOSEN"]
        }
        new_clus_arr.append(new_clus)
    return new_clus_arr


def pick_sentences(cluster_arr, n=2):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for clust in cluster_arr:
        sent_scores = []
        sents = clust["sentences"]
        # words = clust["GROOMED"]
        words_with_counts = clust["groom"]
        # points_dic = {w: len(words) - i for i, w in enumerate(words)}
        points_dic = {w: c for w, c in words_with_counts}
        for i, sen in enumerate(sents):
            sen_split = tokenizer.tokenize(sen)
            # wor = [1 for word in sen_split if word in points_dic]
            wor = [points_dic[word] for word in sen_split if word in points_dic]
            counter = sum(wor)
            sent_scores.append([i, counter])
        sorted_scores = sorted(sent_scores, key=lambda x: x[1], reverse=True)
        to_choose = min(len(sorted_scores), n)
        chosen_sents = []
        for i, count in sorted_scores[0:to_choose]:
            clean = sents[i].replace("'", "")
            clean = clean.replace("\n", "")
            clean = clean.replace("\"", "-")
            chosen_sents.append(clean)
        clust["CHOSEN"] = chosen_sents


def groom_counters(cluster_arr):
    glob_count = collections.Counter()
    for c in cluster_arr:
        count = copy.copy(c["cluster_counter"])
        glob_count.update(count)
    avg_count = collections.Counter()
    per_sent_freq = {}
    total_sentences = sum([a["NUM_SENT"] for a in cluster_arr])
    n_c = len(cluster_arr)
    for key, val in glob_count.items():
        avg_count[key] = int(val / n_c)
        per_sent_freq[key] = float(val / total_sentences)

    for i, clus in enumerate(cluster_arr):
        count = clus["cluster_counter"]
        # count.subtract(avg_count)

        scaled_counts = collections.Counter()
        num_sentences = clus["NUM_SENT"]
        for key, val in per_sent_freq.items():
            scaled_counts[key] = int(val * num_sentences)
        count.subtract(scaled_counts)

        clus["GROOMED"] = [val[0] for val in count.most_common(10) if val[1] != 0]
        clus["groom"] = [val for val in count.most_common(10) if val[1] != 0]
        clus["GROOMED_COUNT"] = [[val[0], clus["cluster_counter"][val[0]]] for val in count.most_common(10) if
                                 val[1] != 0]

    for cl in cluster_arr:
        del cl["cluster_counter"]


def most_common_uncommon(count: collections.Counter, tf_idf_model, n=20):
    srted = sorted([[a, count[a] * tf_idf_model[a]] for a in count.keys()], key=lambda x: x[1], reverse=True)
    return srted[0:min(len(count.keys()), n)]


def closest_to_barycenter(sentences, n=5):
    if len(sentences) == 0:
        return []
    if len(sentences) == 1:
        return [sentences[0][0]]

    bary = sum([s[1] for s in sentences]) / len(sentences)

    def dist(veca, vecb):
        return (sum([(veca[i] - vecb[i]) ** 2 for i in range(len(veca))])) ** .5

    mini = [300000000] * min(n, len(sentences))
    idxs = [0] * min(n, len(sentences))
    for i, sent in enumerate(sentences):
        distance = dist(sent[1], bary)
        if distance < max(mini):
            a = mini.index(max(mini))
            mini[a] = distance
            idxs[a] = sent[0]
    sorted_idxs = sorted([[a, b] for a, b in zip(idxs, mini)], key=lambda x: x[1])
    return [a[0] for a in sorted_idxs]
