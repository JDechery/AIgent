from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

username = 'postgres'
with open('/home/jdechery/.postgrespw.txt','r') as f:
    password = f.readline()[:-1]
# password = ''     # change this
host     = 'localhost'
port     = '5432'            # default port that postgres listens on
db_name  = 'blogs'

engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, blogs))
# if not database_exists(engine.url):
    # create_database(engine.url)

creation_query = '''CREATE TABLE mediumblog (
                    id serial PRIMARY KEY,
                    blog_url TEXT,
                    textcontent TEXT,
                    img_url TEXT,
                    img_path TEXT,
                    title TEXT,
                    claps INTEGER)'''

conn = engine.connect()
conn.execute(creation_query)
conn.close()
