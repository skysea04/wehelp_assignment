from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

BATCH_SIZE = 1000

# PTT
BOARDS = [
    "baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]

CLEANED_FILE = "cleaned_file"
TOKENIZED_FILE = "tokenized_file"

# Tokenizer
IGNORE_TAGS = [
    # word
    "Cab",  # 連接詞，如：等等
    "Cba",  # 連接詞，如：的話
    "P",  # 介詞
    "T",  # 語助詞
    "I",  # 感嘆詞
    # punctuation
    "COLONCATEGORY",  # 冒號
    "COMMACATEGORY",  # 逗號
    "DASHCATEGORY",  # 破折號
    "DOTCATEGORY",  # 點號
    "ETCCATEGORY",  # 刪節號
    "EXCLAMATIONCATEGORY",  # 驚嘆號
    "PARENTHESISCATEGORY",  # 括號
    "PAUSECATEGORY",  # 頓號
    "PERIODCATEGORY",  # 句號
    "QUESTIONCATEGORY",  # 問號
    "SEMICOLONCATEGORY",  # 分號
    "SPCHANGECATEGORY",  # 雙直線
    "WHITESPACE",  # 空白
]
