from flask import render_template
from flask import request
from flaskexample import app
from Mediumrare import db_tools
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
from flaskexample.a_Model import ModelIt
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

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

@app.route('/db')
def birth_page():
    sql_query = """
                SELECT * FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results = pd.read_sql_query(sql_query,conn)
    births = ""
    for i in range(0,10):
        births += query_results.iloc[i]['birth_month']
        births += "<br>"
    return births

@app.route('/db_fancy')
def cesareans_page_fancy():
    sql_query = """
               SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
                """
    query_results=pd.read_sql_query(sql_query,conn)
    births = []
    for i in range(0,query_results.shape[0]):
        births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
    return render_template('cesareans.html',births=births)

@app.route('/input')
def url_input():
    return render_template("input.html", in_url='')

@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  in_url = request.args.get('in_url')
    #just select the Cesareans  from the birth dtabase for the month that the user inputs
  # query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  query = "SELECT title, claps, textcontent, img_path FROM articles WHERE blog_url='%s'" % in_url
  print(query)
  # query_results=pd.read_sql_query(query,conn)
  query_results = conn.execute(query).fetchone()
  print(query_results)
  # births = []
  # for i in range(0,query_results.shape[0]):
      # births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
  the_result = ModelIt(query_results)
  return render_template("output.html", the_result=the_result)
