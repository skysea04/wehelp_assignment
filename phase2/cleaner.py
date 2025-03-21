import re
from threading import Thread

from constants import BOARDS
from logger import stream_log
from utils import ArticleFileManager, CleanedFileManager


def clean_articles_and_append_to_cleaned_file(
    board_name: str, cleaned_file_manager: CleanedFileManager
):
    article_file_manager = ArticleFileManager(board_name=board_name)
    for title, _ in article_file_manager.read_articles():
        title = title.strip()
        title = re.sub(r"^Re:|^Fw:", "", title)
        title = title.strip().lower()

        cleaned_file_manager.write_cleaned_title(board_name, title)

    stream_log.info(f"Finished cleaning board: {board_name}")


if __name__ == "__main__":
    cleaned_file_manager = CleanedFileManager()
    cleaned_file_manager.init_csv_file()

    threads = [
        Thread(
            target=clean_articles_and_append_to_cleaned_file,
            args=(board_name, cleaned_file_manager),
        )
        for board_name in BOARDS
    ]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    stream_log.info("All boards cleaned")
