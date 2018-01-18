from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
# from sqlalchemy.sql import text
# import pandas as pd
# import psycopg2


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
