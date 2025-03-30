import logging
import os
import multiprocessing
from pathlib import Path

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import TokenizedFileManager
from constants import DATA_DIR, TOKENIZED_FILE_FOR_DOC2VEC

CHECK_COUNT = 1000

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_optimal_workers() -> int:
    """Calculate optimal number of workers based on CPU count"""
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of available CPU cores, but at least 4 and at most 8
    optimal_workers = cpu_count // 2
    logger.info(f"CPU count: {cpu_count}, Using {optimal_workers} workers")
    return optimal_workers


def train_doc2vec(file_manager: TokenizedFileManager, vector_size: int, epochs: int, window: int, line_count: int | None = None) -> Doc2Vec:
    model_path_dm = os.path.join("data", f"embedding_model_{vector_size}_{epochs}_{line_count if line_count else 'all'}.d2v")
    corpus_file = os.path.join(DATA_DIR, TOKENIZED_FILE_FOR_DOC2VEC)

    if not os.path.exists(corpus_file):
        with open(corpus_file, "w") as f:
            for line in file_manager.read_titles():
                words = line.split(',')
                f.write(' '.join(words) + '\n')

    logger.info(f"Using corpus file: {corpus_file}")
    
    model_dm = Doc2Vec(
        corpus_file=corpus_file,
        vector_size=vector_size,
        window=window,
        min_count=5,
        workers=get_optimal_workers(),
        epochs=epochs,
        dm=1,
        hs=0,
        negative=15,
    )

    # Save combined model
    model_dm.save(model_path_dm)
    logger.info(f"Model saved to: {model_path_dm}")

    return model_dm

if __name__ == "__main__":
    combinations = [
        # (150, 251, 7, None), 
        # (150, 10, 10, None),
        # (150, 10, 12, None),
        # (150, 10, 15, None),
        (150, 300, 10, None),
        # (50, 10, 25, None),
        # (150, 30, 20, None),
        # (150, 30, 25, None),
        # (150, 100, 25, None),
    ]
    file_manager = TokenizedFileManager()
    result_csv = Path(DATA_DIR, "embedding_models_results.csv")

    # Train and evaluate individual models
    for vector_size, epochs, window, line_count in combinations:
        # train model
        model = train_doc2vec(file_manager, vector_size, epochs, window, line_count)

        # load model
        # model_path = os.path.join("data", f"embedding_model_{vector_size}_{epochs}_{line_count if line_count else 'all'}.d2v")
        # model: Doc2Vec = Doc2Vec.load(model_path)
        first_correct_count = 0
        second_correct_count = 0
        for i, line in enumerate(file_manager.read_titles()):
            new_vector = model.infer_vector(line.split(","))
            similar_docs = model.dv.most_similar([new_vector], topn=2)
            # check if the most similar document is itself
            if similar_docs[0][0] == i:
                first_correct_count += 1
            elif similar_docs[1][0] == i:
                second_correct_count += 1

            if i % 200 == 0:
                logger.info(f"Evaluating progress: {i}/{CHECK_COUNT}")

            if i > CHECK_COUNT:
                break

        self_similarity = first_correct_count / CHECK_COUNT
        second_similarity = (first_correct_count + second_correct_count) / CHECK_COUNT
        logger.info(f"Self-Similarity: {self_similarity}")
        logger.info(f"Second-Similarity: {second_similarity}")

        with open(result_csv, "a") as f:
            f.write(f"{vector_size},{epochs},{line_count or 'all'},{self_similarity},{second_similarity},{window}\n")

