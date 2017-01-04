import json
import logging

import gensim
# import pymysql

import sqlite3

import sys

import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def transfer_data_to_db():
    logging.info("loading model")
    model = gensim.models.Word2Vec.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
    logging.info("model loaded")
    batch_size = 10000
    conn = sqlite3.connect('model.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE words
                 (word text, vec blob)''')
    #
    conn.commit()
    conn.close()
    i = 1
    tups = []

    conn = sqlite3.connect('model.db')
    c = conn.cursor()

    for word, voc in model.vocab.items():
        raw_list = model[word]
        serialized = json.dumps(raw_list.tolist())
        tup = (word, serialized)
        tups.append(tup)
        if i % batch_size == 0:
            c.executemany('INSERT INTO words VALUES (?,?)', tups)
            conn.commit()
            logging.info("inserted in row %s" % i)
            tups = []
        i+=1
    c.executemany('INSERT INTO words VALUES (?,?)', tups)
    conn.commit()
    logging.info("inserted in row %s" % i)
    conn.close()

def inspect():
    conn = sqlite3.connect('model.db')
    c = conn.cursor()
    # c.execute("SELECT COUNT(*) FROM words")
    # print(c.fetchall())
    logging.info("query")
    c.execute("SELECT * FROM words WHERE word='hello'")
    logging.info("query done")
    v = c.fetchall()
    logging.info("fetching done")
    val = v[0][1]
    a = str(val)
    print(type(a))
    logging.info("stringing done")
    vec = json.loads(a)
    print(vec)
    logging.info("json done")
    # model = gensim.models.Word2Vec.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
    # logging.info("MODEL done")
    # vec2 = model["hi"]
    # print(vec2)

    # print("DIFF")
    # print([a-b for a,b in zip(vec,vec2)])
    # print(len(model.vocab.items()))


SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://flask:flaskpassword@flasktest.c5frqpbuteyj.us-west-2.rds.amazonaws.com:3306/flaskdb'


def mysql():
    import pymysql.cursors
    logging.info("loading model")
    model = gensim.models.Word2Vec.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
    logging.info("model loaded")
    batch_size = 1000
    # Connect to the database
    connection = pymysql.connect(host='flasktest.c5frqpbuteyj.us-west-2.rds.amazonaws.com',
                                 user='flask',
                                 password='flaskpassword',
                                 db='flaskdb',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:
            # sql = "DROP table words"
            # cursor.execute(sql)
            # connection.commit()
            # logging.info("DROPPED %s" % cursor.fetchone())
            # sql = """
            #     CREATE TABLE `words` (
            #     `word` varchar(255) COLLATE utf8_bin NOT NULL,
            #     `vec` BLOB NOT NULL,
            #     PRIMARY KEY (`word`)
            #     ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
            #     AUTO_INCREMENT=1 ;
            #     """
            # cursor.execute(sql)
            # connection.commit()
            # logging.info("CREATED %s" % cursor.fetchone())
            tups = []
            i = 1
            for word, voc in model.vocab.items():
                raw_list = model[word]
                # serialized = json.dumps(raw_list.tolist())
                serialized = pickle.dumps(raw_list.tolist())
                tup = (word, serialized)
                tups.append(tup)
                if i % batch_size == 0:
                    # sql = 'INSERT INTO words VALUES (%s,%s)'
                    logging.info("INSERTING %s" % i)
                    sql = 'INSERT INTO `words` (`word`, `vec`) VALUES (%s, %s)'
                    cursor.executemany(sql, tups)
                    connection.commit()
                    logging.info("inserted in row %s" % i)
                    tups = []
                i += 1
            sql = 'INSERT INTO `words` (`word`, `vec`) VALUES (%s, %s)'
            cursor.executemany(sql, tups)
            connection.commit()
            logging.info("inserted in row %s" % i)
            connection.close()
            # Create a new record

            # sql = "DROP TABLE users"
            # sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
            # cursor.execute(sql, ('webmaster@python.org', 'very-secret'))
            # sql = "SELECT * FROM notes"
            # sql = "SELECT * FROM users"
            # cursor.execute(sql)
            # a = cursor.fetchone()
            # print(a)

        # connection is not autocommit by default. So you must commit to save
        # your changes.
        # connection.commit()
        #
        with connection.cursor() as cursor:
        #     # Read a single record
            sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
            cursor.execute(sql, ('webmaster@python.org',))
        #     result = cursor.fetchone()
        #     print(result)
    finally:
        connection.close()


def test():
    import numpy as np
    # import json
    import pymysql.cursors
    logging.info("loading model")
    # model = gensim.models.Word2Vec.load_word2vec_format('../models/GoogleNews-vectors-negative300.bin', binary=True)
    logging.info("model loaded")
    # Connect to the database
    connection = pymysql.connect(host='flasktest.c5frqpbuteyj.us-west-2.rds.amazonaws.com',
                                 user='flask',
                                 password='flaskpassword',
                                 db='flaskdb',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    with connection.cursor() as cursor:
        # sql = "DROP table words"
        sql = """
            CREATE TABLE `words` (
            `word` varchar(255) COLLATE utf8_bin NOT NULL,
            `vec` BLOB NOT NULL,
            PRIMARY KEY (`word`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
            AUTO_INCREMENT=1 ;
            """
        cursor.execute(sql)
        connection.commit()
        # logging.info("BEFORE")
        # sql = "SELECT `word`, `vec` FROM `words` LIMIT 10000"
        # cursor.execute(sql)
        # a = cursor.fetchall()
        # logging.info("AFTER")
        # res_clean = [pickle.loads(ob["vec"]) for ob in a]
        # logging.info("CLEAN")


        # print(len(res_clean))
        # print(a['word'])
        # b = a.get('vec', '')
        # print("SIZE")
        # print(sys.getsizeof(b))
        # print(type(b))
        #
        # pic = pickle.loads(b)
        # print(pic)
        # str_array = str(b)
        # str_array_only = str_array[2:-1]
        # print(str_array_only)
        # c = eval(str_array_only)
        #
        # print(sys.getsizeof(c))
        # p = pickle.dumps(c)
        # print(sys.getsizeof(p))
        # d = json.loads(b)
        # print(d)
        # print("ORIGINAL")
        # d = (model["agent_Bruno_Heiderscheid"])
        # print(d)
        # e = [h-j for h,j in zip(c,d)]
        # print("DIFF")
        # print(e)
    connection.close()

if __name__ == '__main__':
    mysql()
    # test()
    # transfer_data_to_db()
    # inspect()
    # test()