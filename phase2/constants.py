from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

BATCH_SIZE = 1000

BASEBALL = "baseball"
BOY_GIRL = "Boy-Girl"
C_CHAT = "c_chat"
HATE_POLITICS = "hatepolitics"
LIFE_IS_MONEY = "Lifeismoney"
MILITARY = "Military"
PC_SHOPPING = "pc_shopping"
STOCK = "stock"
TECH_JOB = "Tech_Job"

# PTT
BOARDS = [
    BASEBALL,
    BOY_GIRL,
    C_CHAT,
    HATE_POLITICS,
    LIFE_IS_MONEY,
    MILITARY,
    PC_SHOPPING,
    STOCK,
    TECH_JOB,
]

# BOARD_TAG_MAP = {
#     BASEBALL: 1,
#     BOY_GIRL: 2,
#     C_CHAT: 3,
#     HATE_POLITICS: 4,
#     LIFE_IS_MONEY: 5,
#     MILITARY: 6,
#     PC_SHOPPING: 7,
#     STOCK: 8,
#     TECH_JOB: 9,
# }

CLEANED_FILE = "cleaned_file"
TOKENIZED_FILE = "tokenized_file"
EMBEDDING_RESULT_FILE = "embedding_result"
TOKENIZED_FILE_FOR_DOC2VEC = "tokenzied_file_for_doc2vec.txt"


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
