import logging
import os
from itertools import product

from gensim.models.doc2vec import Doc2Vec
from utils import TokenizedFileManager, EmbeddingResultFileManager, get_optimal_workers
from constants import DATA_DIR, TOKENIZED_FILE_FOR_DOC2VEC

CHECK_COUNT = 2000

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def train_doc2vec(
    file_manager: TokenizedFileManager,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    hs: int,
    negative: int,
    sample: float,
) -> Doc2Vec:
    model_path_dm = os.path.join(
        "data",
        f"embedding_model_{vector_size}_{window}_{min_count}_{epochs}_{hs}_{negative}_{sample}.d2v",
    )
    corpus_file = os.path.join(DATA_DIR, TOKENIZED_FILE_FOR_DOC2VEC)

    if not os.path.exists(corpus_file):
        with open(corpus_file, "w") as f:
            for line in file_manager.read_titles():
                words = line.split(",")[1:]
                f.write(" ".join(words) + "\n")

    logger.info(f"Using corpus file: {corpus_file}")

    model_dm = Doc2Vec(
        corpus_file=corpus_file,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=get_optimal_workers(),
        epochs=epochs,
        dm=1,
        hs=hs,
        negative=negative,
        sample=sample,
    )

    # Save combined model
    model_dm.save(model_path_dm)
    logger.info(f"Model saved to: {model_path_dm}")

    return model_dm


if __name__ == "__main__":
    tokenized_file_manager = TokenizedFileManager()
    result_file_manager = EmbeddingResultFileManager()
    result_file_manager.init_file()

    vector_sizes = [300]
    windows = [2]
    min_counts = [5]
    epochs_lst = [10]
    hs_negative = [(1, 0)]
    sample = [0]

    # Create all possible combinations
    combinations = list(
        product(vector_sizes, windows, min_counts, epochs_lst, hs_negative, sample)
    )
    # logger.info(f"Total combinations to try: {len(combinations)}")
    # print(combinations[:10])

    # Train and evaluate individual models
    for vector_size, window, min_count, epochs, (hs, negative), sample in combinations:
        # train model
        model = train_doc2vec(
            tokenized_file_manager,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            hs=hs,
            negative=negative,
            sample=sample,
        )

        # load model
        first_correct_count = 0
        second_correct_count = 0
        for i, line in enumerate(tokenized_file_manager.read_titles()):
            new_vector = model.infer_vector(line.split(",")[1:])
            similar_docs = model.dv.most_similar([new_vector], topn=2)
            # check if the most similar document is itself
            if similar_docs[0][0] == i:
                first_correct_count += 1
            elif similar_docs[1][0] == i:
                second_correct_count += 1

            if i % 200 == 0 and i > 0:
                logger.info(f"Evaluating progress: {i}/{CHECK_COUNT}")

            if i > CHECK_COUNT:
                break

        self_similarity = round(first_correct_count / CHECK_COUNT, 3)
        second_similarity = round(
            (first_correct_count + second_correct_count) / CHECK_COUNT, 3
        )
        logger.info(f"Self-Similarity: {self_similarity}")
        logger.info(f"Second-Similarity: {second_similarity}")

        result_file_manager.write_embedding_result(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            hs=hs,
            negative=negative,
            sample=sample,
            self_similarity=self_similarity,
            second_similarity=second_similarity,
        )
