import logging
import os

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import TokenizedFileManager

EMBEDDING_MODEL_PATH = os.path.join("data", "embedding_model_50_50.d2v")
CHECK_COUNT = 100

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def train_doc2vec(file_manager: TokenizedFileManager) -> Doc2Vec:
    tagged_data = [
        TaggedDocument(words=line.split(","), tags=[i])
        for i, line in enumerate(file_manager.read_titles())
    ]

    model = Doc2Vec(
        vector_size=50,
        window=3,
        min_count=2,
        workers=8,
        epochs=50,
        dm=1,
        hs=0,
        negative=10,
    )

    # build vocabulary
    model.build_vocab(tagged_data)

    # train model
    logger.info("Start training Doc2Vec model...")
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # save model
    model.save(EMBEDDING_MODEL_PATH)
    logger.info(f"Model saved to: {EMBEDDING_MODEL_PATH}")

    return model


if __name__ == "__main__":
    file_manager = TokenizedFileManager()

    # train model
    model = train_doc2vec(file_manager)

    # load model
    # model = Doc2Vec.load(EMBEDDING_MODEL_PATH)
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

        if i > CHECK_COUNT:
            break

    logger.info(f"Self-Similarity: {first_correct_count / CHECK_COUNT}")
    logger.info(
        f"Second-Similarity: {(first_correct_count + second_correct_count) / CHECK_COUNT}"
    )
