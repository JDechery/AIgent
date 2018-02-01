from flask import render_template
from flask import request, redirect
from flaskexample import app
from Mediumrare import db_tools, gensim_nlp, predictor_model
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
from flaskexample.a_Model import ModelBetter
import pandas as pd
import psycopg2

#user = 'postgres' #add your username here (same as previous postgreSQL)
#host = 'localhost'
#dbname = 'birth_db'
# user = 'postgres'
# password = 'dextraN17'     # change this
# host     = 'localhost'
# port     = '5432'            # default port that postgres listens on
# dbname  = 'birth_db'
# engine = create_engine( 'postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname) )
#db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# conn = None
# conn = psycopg2.connect(database = dbname, user = user, host=host, password=password)
conn = db_tools.get_conn()
embed_fname = '../doc2vec.model'
embedder = gensim_nlp.DocEmbedder()
embedder.load_model(fname=embed_fname)

clf_fname = '../forest_classifier.pkl'
clf,_,_ = predictor_model.load_classifier()

# @app.route('/db')
# def birth_page():
#     sql_query = """
#                 SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';
#                 """
#     query_results = pd.read_sql_query(sql_query,conn)
#     births = ""
#     for i in range(0,10):
#         births += query_results.iloc[i]['birth_month']
#         births += "<br>"
#     return births

# @app.route('/db_fancy')
# def cesareans_page_fancy():
#     sql_query = """
#                SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
#                 """
#     query_results=pd.read_sql_query(sql_query,conn)
#     births = []
#     for i in range(0,query_results.shape[0]):
#         births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#     return render_template('cesareans.html',births=births)
# @app.route('/')
@app.route('/')
def home():
    return redirect('/input', code=301)

@app.route('/about')
def index():
    return render_template("about.html",
    title = 'About Me', user = { 'nickname': 'Joe' })

@app.route('/slides')
def slides():
    return render_template("slides.html", title="slides")

@app.route('/input', methods=['GET','POST'])
def url_input():
    return render_template("input.html", blogtext='')

@app.route('/output', methods=['GET'])
def cesareans_output():
  blogtext = request.args.get('blogtext')
  print(blogtext)
  # channel_rec = ModelIt(blogtext, embedder.model, clf)
  channel_rec = ModelBetter(blogtext, embedder.model, clf)
  # print(channel_rec)
  return render_template("output_interactive.html", channel_rec=channel_rec)
