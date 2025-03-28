import re
import time
from threading import Thread
from typing import Callable

import requests
from bs4 import BeautifulSoup

from constants import BOARDS
from logger import file_log, stream_log
from utils import ArticleFileManager

BASE_URL = "https://www.ptt.cc"


class CrawlPttException(Exception):
    pass


def get_articles(soup: BeautifulSoup) -> list[tuple[str, str]]:
    articles = []
    for item in soup.select(".r-ent"):
        title_tag = item.select_one(".title a")
        if title_tag:
            title = title_tag.text.strip()
            link = f"{BASE_URL}{title_tag['href']}"
            articles.append((title, link))

    return articles


def get_last_page_number(soup: BeautifulSoup) -> int | None:
    next_page_buttons = soup.select(".btn-group-paging a")
    for btn in next_page_buttons:
        if "‹ 上頁" in btn.text:
            match = re.search(r"index(\d+)\.html", btn["href"])
            if match:
                return int(match.group(1))

    return None


def retry(func: Callable, max_retries: int = 3, delay: int = 3):
    def wrapper(*args, **kwargs):
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                stream_log.error(f"Request Error: {e}")

                if i == max_retries - 1:
                    raise CrawlPttException(e)

                time.sleep(delay)

    return wrapper


@retry
def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, cookies={"over18": "1"}, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    file_log.info(f"Crawled {url}")

    return soup


def crawl_ptt(board_name: str, max_article_cnt: int = 200000) -> None:
    url = f"{BASE_URL}/bbs/{board_name}/index.html"
    first_page = True
    cur_article_cnt = 0
    article_file_manager = ArticleFileManager(board_name=board_name)
    article_file_manager.init_csv_file()
    page_number = float("inf")

    while cur_article_cnt < max_article_cnt:
        try:
            soup = get_soup(url)
        except CrawlPttException:
            page_number -= 1
            if page_number < 1:
                break

            url = f"{BASE_URL}/bbs/{board_name}/index{page_number}.html"
            continue

        articles = get_articles(soup)
        cur_article_cnt += len(articles)
        article_file_manager.write_articles(articles)

        if first_page:
            first_page = False
            page_number = get_last_page_number(soup)
            if not page_number:
                break

            url = f"{BASE_URL}/bbs/{board_name}/index{page_number}.html"

        else:
            page_number -= 1
            if page_number < 1:
                break

        url = f"{BASE_URL}/bbs/{board_name}/index{page_number}.html"

    stream_log.info(f"Crawled {cur_article_cnt} articles from {board_name}")


if __name__ == "__main__":
    threads = [Thread(target=crawl_ptt, args=(board_name,)) for board_name in BOARDS]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    stream_log.info("All boards crawled")
