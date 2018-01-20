# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.sql import text
# import pandas as pd
# import psycopg2
# import pudb
from Mediumrare import db_tools

class TutorialPipeline(object):
    def open_spider(self, spider):
        # print('starting connection')
        # username = 'postgres'
        # with open('/home/jdechery/.postgrespw.txt','r') as f:
        #     password = f.readline()[:-1]
        # # password = ''     # change this
        # host     = 'localhost'
        # port     = '5432'            # default port that postgres listens on
        # db_name  = 'blogs'
        # engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name))
        # self.conn = engine.connect()
        self.conn = db_tools.get_conn()
        self.query = text("""INSERT into mediumblogfull
                          (blog_url, textcontent,
                          img_url, img_path, title, claps,
                          author, pub_date, tags, channel)
                          VALUES (:blog_url, :textcontent,
                          :img_url, :img_path, :title, :claps,
                          :author, :pub_date, :tags, :channel)""")
        # print('db connection created')

    def close_spider(self, spider):
        # print(self.conn.execute('select count(*) from blogs'))
        # self.conn.commit()
        self.conn.close()
        # print('db connection close successfully')

    def process_item(self, item, spider):
        # for val in item.keys():
        #     if not isinstance(val,int):
        #         val = text(val)

        with self.conn.begin() as trans:
            # print('item values:', item.values())
            # pudb.set_trace()
            self.conn.execute(self.query, **item)
            trans.commit()

        return item
