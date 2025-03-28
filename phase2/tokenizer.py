from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter

from constants import BATCH_SIZE, IGNORE_TAGS
from logger import stream_log
from utils import CleanedFileManager, TokenizedFileManager

stream_log.info("Initializing drivers ... WS")
ws_driver = CkipWordSegmenter(model="bert-base")
stream_log.info("Initializing drivers ... POS")
pos_driver = CkipPosTagger(model="bert-base")
stream_log.info("Initializing drivers ... done")


def tokenize(board_names: list[str], texts: list[str]) -> list[str]:
    ws = ws_driver(texts)
    pos = pos_driver(ws)

    tokenized_texts = []
    for board_name, text_ws, text_pos in zip(board_names, ws, pos):
        tokenized_texts.append(
            f"{board_name}, {convert_ws_pos_sentence(text_ws, text_pos)}"
        )

    return tokenized_texts


def convert_ws_pos_sentence(text_ws: list[str], text_pos: list[str]) -> str:
    res = []
    for word_ws, word_pos in zip(text_ws, text_pos):
        if word_pos in IGNORE_TAGS:
            continue

        res.append(word_ws)

    return ", ".join(res)


if __name__ == "__main__":
    cleaned_file_manager = CleanedFileManager()
    tokenized_file_manager = TokenizedFileManager()
    tokenized_file_manager.init_file()

    board_names, texts = [], []
    for board_name, title in cleaned_file_manager.read_titles():
        board_names.append(board_name)
        texts.append(title)

        if len(board_names) == BATCH_SIZE:
            tokenized_texts = tokenize(board_names, texts)
            tokenized_file_manager.write_tokenized_title(tokenized_texts)
            board_names, texts = [], []

    tokenized_texts = tokenize(board_names, texts)
    tokenized_file_manager.write_tokenized_title(tokenized_texts)
