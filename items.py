# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class TopicItem(scrapy.Item):
    topic_id = scrapy.Field()
    heading = scrapy.Field()
    datetime = scrapy.Field()
    location = scrapy.Field()
    username = scrapy.Field()
    topic_desc = scrapy.Field()


class ReplyItem(scrapy.Item):
    reply_id = scrapy.Field()
    topic_id = scrapy.Field()
    reply_content = scrapy.Field()
    replier_username = scrapy.Field()
    replier_location = scrapy.Field()
    reply_datetime = scrapy.Field()
