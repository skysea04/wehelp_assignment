import random
import os
from gensim.models.doc2vec import Doc2Vec
from logger import stream_log
from utils import TokenizedFileManager

CHECK_COUNT = 1000

# load model
model_path = os.path.join("data", "embedding_model_100_2_5_15_1_0_0.1.d2v")
model: Doc2Vec = Doc2Vec.load(model_path)

file_manager = TokenizedFileManager()
processed_count = 0
first_correct_count, second_correct_count = 0, 0
for i, line in enumerate(file_manager.read_titles()):
    if random.random() > 0.7:
        continue

    processed_count += 1

    new_vector = model.infer_vector(line.split(",")[1:])
    similar_docs = model.dv.most_similar([new_vector], topn=2)

    # check if the most similar document is itself
    if similar_docs[0][0] == i:
        first_correct_count += 1
    elif similar_docs[1][0] == i:
        second_correct_count += 1

    if processed_count % 200 == 0:
        stream_log.info(f"Evaluating progress: {processed_count}/{CHECK_COUNT}")

    if processed_count > CHECK_COUNT:
        break

self_similarity = first_correct_count / CHECK_COUNT
second_similarity = (first_correct_count + second_correct_count) / CHECK_COUNT
stream_log.info(f"Self-Similarity: {self_similarity}")
stream_log.info(f"Second-Similarity: {second_similarity}")
