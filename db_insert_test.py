from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.sql import text

username = 'postgres'
with open('/home/jdechery/.postgrespw.txt','r') as f:
    password = f.readline()[:-1]
# password = ''     # change this
host     = 'localhost'
port     = '5432'            # default port that postgres listens on
db_name  = 'blogs'
engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name))
if not database_exists(engine.url):
    create_database(engine.url)
query = text("INSERT into mediumblog (blog_url, textcontent, img_url, img_path, title, claps) values (:blog_url, :textcontent, :img_url, :img_path, :title, :claps)")

test_vals = {'blog_url':'www', 'img_url':'qwerty', 'title':'hello world', 'claps':3, 'textcontent':'asdf', 'img_path':'/home'}
# print(test_vals.keys())
conn = engine.connect()
with conn.begin() as trans:
    conn.execute(query, test_vals)
    trans.commit()
conn.close()
