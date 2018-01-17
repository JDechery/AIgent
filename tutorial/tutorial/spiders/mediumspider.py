import scrapy
from tutorial.items import BlogPost
from scrapy.selector import Selector
import requests
from itertools import product

def set_resolution(img_url, res=800):
    parts = img_url.split('/')
    res_idx = parts.index('max')+1
    parts[res_idx] = str(res)
    return '/'.join(parts)

def save_image(img_url, img_path):
    success = False
    with open(img_path,'wb') as f:
        response = requests.get(img_url)
        if response.status_code == 200:
            f.write(requests.get(img_url).content)
            success = True
    return success

def generate_archive_url(channels, years, months):
    base_url = 'https://www.medium.com/{}/archive/{}/{}'
    for (channel, yr, mo) in product(channels, years, months):
        yield base_url.format(channel, yr, mo)

class MediumSpider(scrapy.Spider):
    name = "mediumspider"

    def start_requests(self):
        # channels = ['backchannel', 'matter', 'the-mission']
        # years = ['2013', '2014', '2015', '2016', '2017']
        channels = ['matter']
        years = ['2017']
        months = ['%02d'%x for x in range(3,13)]
        # urls = ['https://writingcooperative.com/stop-romanticizing-your-writing-career-8d1de425a54b']

        for url in generate_archive_url(channels, years, months):
            yield scrapy.Request(url=url,
                                 meta = {
                                    'dont_redirect': True,
                                    'handle_httpstatus_list': [302]},
                                 callback=self.parse)

    def parse(self, response):
        post_urls = response.xpath("//div[contains(@class, 'postArticle-readMore')]/a/@href").extract()
        for url in post_urls:
            raw_url = url.split('?')[0]
            yield scrapy.Request(url=url, callback=self.parse_blog_data)

    def parse_blog_data(self, response):
        blogdata = BlogPost()

        blogdata['blog_url'] = response.url.split('?')[0]

        textcontent = response.xpath("//p/text()").extract()
        textcontent = '\n'.join(textcontent)
        blogdata['textcontent'] = textcontent

        # blogdata['title'] = response.xpath("//h1[contains(@class, 'title')]/text()").extract_first()
        title = response.xpath('//title/text()').extract_first()
        blogdata['title'] = title.split('–')[0]

        claps = response.xpath("//button[contains(@data-action, 'show-recommends')]/text()").extract_first()
        blogdata['claps'] = int(claps.replace('.','').replace('K','000'))

        img_url = response.xpath('//div/img/@src').extract_first()
        img_url = set_resolution(img_url)
        blogdata['img_url'] = img_url

        img_path = '/home/jdechery/Pictures/medium/' + blogdata['title'] + '.jpg'
        img_save_success = save_image(img_url, img_path)
        if img_save_success:
            self.log('img saved successfully')
        else:
            self.log('img not saved')
        blogdata['img_path'] = img_path

        self.log(f'collected blog post {response.url}')
        return blogdata
