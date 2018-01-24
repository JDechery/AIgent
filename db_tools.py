from sqlalchemy import create_engine
from sqlalchemy.sql import text
import psycopg2
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# %%
def get_conn():
    username = 'postgres'
    with open('/home/jdechery/.postgrespw.txt','r') as f:
        password = f.readline()[:-1]
    host     = 'localhost'
    port     = '5432'
    db_name  = 'blogs'
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name))
    conn = engine.connect()
    return conn

# %% one and done
def create_clean_table(conn=get_conn()):
    colnames = ('id', 'blog_url', 'rawtext', 'img_url', 'img_path', 'title', 'claps')
    creation_query = """CREATE TABLE mediumclean (
                        id serial PRIMARY KEY,
                        blog_url TEXT,
                        rawtext TEXT,
                        cleantext TEXT,
                        img_url TEXT,
                        img_path TEXT,
                        title TEXT,
                        claps INTEGER)"""

    # conn = engine.connect()
    conn.execute(creation_query)

    select_query = """SELECT * FROM mediumblog"""
    insert_query = text("""INSERT INTO mediumclean (blog_url, rawtext, cleantext, img_url, img_path, title, claps)
                           VALUES (:blog_url, :rawtext, :cleantext, :img_url, :img_path, :title, :claps)""")

    select_out = conn.execute(select_query)
    current_row = select_out.fetchone()
    while current_row is not None:
        # new_row = current_row.copy()
        new_row = dict(zip(colnames, current_row))
        new_row['cleantext'] = clean_document(new_row['rawtext'])
        conn.execute(insert_query, new_row)
        current_row = select_out.fetchone()
    conn.close()

def create_clean_full(conn=get_conn()):
    old_cols = ('id', 'blog_url', 'textcontent', 'textstructure', 'author',
                'pub_date', 'tags', 'img_url', 'img_path', 'title', 'claps')
    new_cols = ('id', 'blog_url', 'rawtext', 'textstructure', 'author',
                'pub_date', 'tags', 'cleantext', 'img_url', 'img_path', 'title', 'claps')

    creation_query = """CREATE TABLE mediumcleanfull (
                        id serial PRIMARY KEY,
                        blog_url TEXT,
                        rawtext TEXT,
                        textstructure TEXT,
                        author TEXT,
                        pub_date TEXT,
                        tags TEXT,
                        cleantext TEXT,
                        img_url TEXT,
                        img_path TEXT,
                        title TEXT,
                        claps INTEGER)"""

    conn.execute(creation_query)

    select_query = "SELECT " + ", ".join(old_cols) + " FROM mediumblogfull ORDER BY id"
    insert_query = text("""INSERT INTO mediumcleanfull
                        (blog_url, rawtext, cleantext, textstructure, author,
                        pub_date, tags, img_url, img_path, title, claps)
                        VALUES
                        (:blog_url, :rawtext, :cleantext, :textstructure, :author,
                        :pub_date, :tags, :img_url, :img_path, :title, :claps)""")

    select_out = conn.execute(select_query)
    current_row = select_out.fetchone()
    while current_row is not None:
        # new_row = current_row.copy()
        new_row = dict(zip(old_cols, current_row))
        new_row['rawtext'] = new_row['textcontent']
        new_row['cleantext'] = clean_document(new_row['rawtext'])
        new_row.pop('textcontent')
        conn.execute(insert_query, new_row)
        current_row = select_out.fetchone()
    conn.close()
    # return current_row, new_row
# %%
def clean_document(rawtext):
    ltzr = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    cleantext = gensim.utils.simple_preprocess(rawtext)
    cleantext = [token for token in cleantext if token not in stop_words]
    cleantext = [ltzr.lemmatize(ltzr.lemmatize(token, 'n'),'v') for token in cleantext]
    cleantext = ' '.join(cleantext)
    return cleantext

# one and done
def create_raw_table(conn=get_conn()):
    creation_query = """CREATE TABLE mediumblog (
                        id serial PRIMARY KEY,
                        blog_url TEXT,
                        textcontent TEXT,
                        img_url TEXT,
                        img_path TEXT,
                        title TEXT,
                        claps INTEGER)"""

    conn = engine.connect()
    conn.execute(creation_query)
    conn.close()

# %% one and done
def create_full_table(conn=get_conn()):
    creation_query = """CREATE TABLE mediumblogfull (
                        id serial PRIMARY KEY,
                        blog_url TEXT,
                        textcontent TEXT,
                        textstructure TEXT,
                        img_url TEXT,
                        img_path TEXT,
                        title TEXT,
                        author TEXT,
                        pub_date TEXT,
                        tags TEXT,
                        channel TEXT,
                        claps INTEGER)"""
    conn.execute(creation_query)
    conn.close()
# create_full_table()
