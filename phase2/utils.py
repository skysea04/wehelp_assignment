import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from constants import CLEANED_FILE, DATA_DIR, TOKENIZED_FILE


@dataclass
class ArticleFileManager:
    board_name: str

    @property
    def csv_file_path(self) -> Path:
        return DATA_DIR / f"{self.board_name}.csv"

    def init_csv_file(self):
        with open(self.csv_file_path, "w") as f:
            f.write("title,url\n")

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
        with open(self.csv_file_path, "w") as f:
            f.write("board_name,title\n")

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
        Path(self.file_path).touch()

    def write_tokenized_title(self, tokenized_titles: list[str]):
        with open(self.file_path, "a") as f:
            for tokenized_title in tokenized_titles:
                f.write(f"{tokenized_title}\n")

    def read_titles(self) -> Generator[str, None, None]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()
