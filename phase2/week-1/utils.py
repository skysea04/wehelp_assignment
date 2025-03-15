import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from constants import DATA_DIR


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
        return DATA_DIR / "cleaned_file.csv"

    def init_csv_file(self):
        with open(self.csv_file_path, "w") as f:
            f.write("board_name,title\n")

    def write_cleaned_title(self, board_name: str, title: str):
        with open(self.csv_file_path, "a") as f:
            f.write(f"{board_name},{title}\n")
