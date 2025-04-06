import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import time
from functools import wraps
from logger import stream_log
import multiprocessing

from constants import CLEANED_FILE, DATA_DIR, TOKENIZED_FILE, EMBEDDING_RESULT_FILE


@dataclass
class ArticleFileManager:
    board_name: str

    @property
    def csv_file_path(self) -> Path:
        return DATA_DIR / f"{self.board_name}.csv"

    def init_csv_file(self):
        if not Path(self.csv_file_path).exists():
            file = Path(self.csv_file_path)
            file.write_text("title,url\n")

    def read_articles(self) -> Generator[tuple[str, str], None, None]:
        with open(self.csv_file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row["title"], row["url"]

    def write_articles(self, articles: list[tuple[str, str]]):
        articles_str = "\n".join([f"{title},{link}" for title, link in articles])

        with open(self.csv_file_path, "a") as f:
            f.write(articles_str + "\n")

    def write_cleaned_title(self, title: str):
        with open(self.csv_file_path, "a") as f:
            f.write(title + "\n")


@dataclass
class CleanedFileManager:
    @property
    def csv_file_path(self) -> Path:
        return DATA_DIR / f"{CLEANED_FILE}.csv"

    def init_csv_file(self):
        if not Path(self.csv_file_path).exists():
            file = Path(self.csv_file_path)
            file.write_text("board_name,title\n")

    def read_titles(self) -> Generator[tuple[str, str], None, None]:
        with open(self.csv_file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row["board_name"], row["title"]

    def write_cleaned_title(self, board_name: str, title: str):
        with open(self.csv_file_path, "a") as f:
            f.write(f"{board_name},{title}\n")


@dataclass
class TokenizedFileManager:
    @property
    def file_path(self) -> Path:
        return DATA_DIR / f"{TOKENIZED_FILE}.txt"

    def init_file(self):
        if not Path(self.file_path).exists():
            Path(self.file_path).touch()

    def write_tokenized_title(self, tokenized_titles: list[str]):
        with open(self.file_path, "a") as f:
            for tokenized_title in tokenized_titles:
                f.write(f"{tokenized_title}\n")

    def read_titles(self) -> Generator[str, None, None]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()


@dataclass
class EmbeddingResultFileManager:
    @property
    def file_path(self) -> Path:
        return DATA_DIR / f"{EMBEDDING_RESULT_FILE}.csv"

    def init_file(self):
        if not Path(self.file_path).exists():
            file = Path(self.file_path)
            file.write_text(
                "vector_size,window,min_count,epochs,hs,negative,sample,self_similarity,second_similarity\n"
            )

    def write_embedding_result(
        self,
        vector_size: int,
        window: int,
        min_count: int,
        epochs: int,
        hs: int,
        negative: int,
        sample: float,
        self_similarity: float,
        second_similarity: float,
    ):
        with open(self.file_path, "a") as f:
            f.write(
                f"{vector_size},{window},{min_count},{epochs},{hs},{negative},{sample},{self_similarity},{second_similarity}\n"
            )


def get_optimal_workers() -> int:
    """Calculate optimal number of workers based on CPU count"""
    cpu_count = multiprocessing.cpu_count()
    optimal_workers = cpu_count // 2
    stream_log.info(f"CPU count: {cpu_count}, Using {optimal_workers} workers")
    return optimal_workers


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        stream_log.info(
            f"Function {func.__name__} took {end_time - start_time} seconds"
        )
        return result

    return wrapper
