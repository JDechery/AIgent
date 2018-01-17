# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class BlogPost(scrapy.Item):
    title = scrapy.Field()
    textcontent = scrapy.Field()
    claps = scrapy.Field()
    img_url = scrapy.Field()
    blog_url = scrapy.Field()
    img_path = scrapy.Field()

class TutorialItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass
