# -*- coding: utf-8 -*-
import scrapy
import bs4
from hkia_spider.items import TopicItem, ReplyItem


class TripAdvisorSpider(scrapy.Spider):
    name = 'trip_advisor'
    start_urls = [
        'https://en.tripadvisor.com.hk/SearchForums?q=hk+airport&s='
    ]
    topic_id = 0

    def parse(self, response):
        # Get list of topics selecting by using multiple classes
        topics_list = response.css('.topofgroup.topofgroup')
        print '\n\n####################### New Topics List #######################\n\n'

        for topic in topics_list:
            # Get first of topic and it's reference attribute
            topic_link = topic.css(
                ':nth-child(1)').css('a::attr(href)').extract_first()

            yield response.follow(topic_link, callback=self.parse_topic_page)

        # Next Page
        page_links = response.css('.pagination a::attr(href)').extract()
        if len(page_links) > 0:
            yield response.follow(page_links[-1], callback=self.parse)

    def parse_topic_page(self, response):
        
        topic_item = TopicItem()

        topic_id = self.topic_id
        self.topic_id += 1
        topic_item['topic_id'] = topic_id

        
        topic_item['heading'] = response.css('#HEADING::text').extract_first()
        topic_item['datetime'] = response.css(
            '.bx01 .postDate::text').extract_first()
        
        topic_item['location'] = response.css(
            '.bx01 .location::text').extract_first()

        topic_item['username'] = response.css(
            '.bx01 .username span::text').extract_first()

        post_soup = bs4.BeautifulSoup(
            response.css('.bx01').extract_first(), 'lxml')

        if post_soup.script:
            post_soup.script.decompose()
        topic_item['topic_desc'] = '\n'.join(
            [p.text for p in post_soup.find_all('p')])

        yield topic_item

        # Replies
        replies = response.css('.balance > .post').extract()

        reply_id = 0
        for r in replies:

            reply_item = ReplyItem()

            reply_item['topic_id'] = topic_id

            reply_item['reply_id'] = reply_id
            reply_id += 1

            r_soup = bs4.BeautifulSoup(r, 'lxml')
            if r_soup.script:
                r_soup.script.decompose()
            reply_item['reply_content'] = '\n'.join(
                [p.text for p in r_soup.find('div', class_='postBody').find_all('p')])

            replier_username_div = r_soup.find('div', class_='username')
            if replier_username_div:
                replier_username = replier_username_div.find('span').text
            else:
                replier_username = None
            reply_item['replier_username'] = replier_username

            replier_location_div = r_soup.find('div', class_='location')
            if replier_location_div:
                replier_location = replier_location_div.text
            else:
                replier_location = None
            reply_item['replier_location'] = replier_location

            reply_datetime = r_soup.find('div', class_='postDate').text
            reply_item['reply_datetime'] = reply_datetime

            yield reply_item

        # Reply pages
        next_page_link = response.css(
            '.guiArw.sprite-pageNext::attr(href)').extract_first()

        if next_page_link:
            yield response.follow(next_page_link, callback=self.parse_next_reply_page, meta={'topic_id': topic_id,
                                                                                             'reply_id': reply_id})

        print '\n############## Done with Topic: {} ##############\n'.format(topic_id)

    def parse_next_reply_page(self, response):
        topic_id = response.meta.get('topic_id')
        reply_id = response.meta.get('reply_id')

        replies = response.css('.balance > .post').extract()

        for r in replies:

            reply_item = ReplyItem()

            reply_item['topic_id'] = topic_id

            reply_item['reply_id'] = reply_id
            reply_id += 1

            r_soup = bs4.BeautifulSoup(r, 'lxml')
            if r_soup.script:
                r_soup.script.decompose()
            reply_item['reply_content'] = '\n'.join(
                [p.text for p in r_soup.find('div', class_='postBody').find_all('p')])

            replier_username_div = r_soup.find('div', class_='username')
            if replier_username_div:
                replier_username = replier_username_div.find('span').text
            else:
                replier_username = None
            reply_item['replier_username'] = replier_username

            replier_location_div = r_soup.find('div', class_='location')
            if replier_location_div:
                replier_location = replier_location_div.text
            else:
                replier_location = None
            reply_item['replier_location'] = replier_location

            reply_datetime = r_soup.find('div', class_='postDate').text
            reply_item['reply_datetime'] = reply_datetime

            yield reply_item

        next_page_link = response.css(
            '.guiArw.sprite-pageNext a::attr(href)').extract_first()

        if next_page_link:
            yield response.follow(next_page_link, callback=self.parse_next_reply_page, meta={'topic_id': topic_id,
                                                                                             'reply_id': reply_id})
